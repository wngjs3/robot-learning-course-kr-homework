"""기록된 원격 조종(teleop) .zarr 데이터셋을 읽고 모방 학습을 위한 액션을 계산합니다.

액션은 연속된 상태 사이의 상대적인 변화량(delta)으로 정의됩니다:
    a_t = s_{t+1} - s_{t}
각 에피소드의 마지막 타임스텝은 제외됩니다 (참조할 미래 상태가 없기 때문).

그리퍼의 경우, 큐브에 힘을 가하기 위해 그리퍼를 더 꽉 닫아야 하며, 이는 상태가 아닌 제어 입력에서만 기록될 수 있으므로 원격 조종 중 기록된 제어 명령을 액션으로 사용합니다.

세 가지 액션 공간을 지원합니다 (--action-space로 선택):

  ee       - EE xyz 위치 변화량 (3차원)
  ee_full  - EE 전체 포즈 변화량: delta_pos(3) + delta_euler(3) (6차원)
  joints   - Jaw를 제외한 관절 각도 변화량 (5차원)

그리퍼 액션은 모든 모드에서 별도의 ``action_gripper`` 배열로 저장됩니다.

사용 예시:
    python scripts/compute_actions.py --action-space ee
    python scripts/compute_actions.py --action-space ee_full
    python scripts/compute_actions.py --action-space joints
    python scripts/compute_actions.py --action-space joints --datasets-dir ./datasets/raw/multi_cube
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr

# ── 쿼터니언 헬퍼 함수 (wxyz 컨벤션) ──────────────────────────────


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """wxyz 포맷의 단위 쿼터니언의 켤레(conjugate)를 구합니다 (단위 쿼터니언의 역원과 동일)."""
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """wxyz 포맷의 두 쿼터니언 배열의 해밀턴 곱(Hamilton product)을 구합니다."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """wxyz 쿼터니언을 오일러 각도(roll, pitch, yaw)로 변환합니다."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # 롤 (x축)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # 피치 (y축)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # 요 (z축)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=-1)


def _ee_full_delta(s_curr: np.ndarray, s_next: np.ndarray) -> np.ndarray:
    """ee_full 액션을 계산합니다: delta_pos(3) + delta_euler(3).

    상태(States) 형태는 (L, 7): [x, y, z, qw, qx, qy, qz].
    반환(Returns) 형태는 (L, 6): [dx, dy, dz, droll, dpitch, dyaw].
    
    """
    pos_delta = s_next[:, :3] - s_curr[:, :3]
    q_rel = quat_multiply(s_next[:, 3:], quat_conjugate(s_curr[:, 3:]))
    euler_delta = quat_to_euler(q_rel)
    return np.concatenate([pos_delta, euler_delta], axis=-1)


# ── 액션 공간 명명 설정 (수정 금지) ────────────────────────
# CLI 선택 → (action_label, state_label, 출력 배열용 키 접미사) 매핑
_ACTION_SPACE_LABELS: dict[str, tuple[str, str, str]] = {
    "ee": ("ee_pos_xyz(3)", "ee_pos_xyz(3)", "ee_xyz"),
    "ee_full": ("delta_pos(3)+delta_euler(3)", "ee_pos(3)+quat_wxyz(4)", "ee_full"),
    "joints": ("joint_angles(5)", "joint_angles(5)", "joints"),
}


def select_action_space(
    action_space: str, merged: dict[str, np.ndarray]
) -> tuple[np.ndarray, str, str, str]:
    """선택한 액션 공간에 맞는 상태 배열을 선택하고 슬라이싱합니다.

    Parameters
    ----------
    action_space : str
        ``"ee"``, ``"ee_full"``, ``"joints"`` 중 하나.
    merged : dict
        :func:`load_and_merge_zarrs`에서 병합된 데이터. 관련 키는
        [x, y, z, qw, qx, qy, qz] 열을 가진 ``"state_ee"`` (N, 7) 및
        각 관절 각도 열을 가진 ``"state_joints"`` (N, 6)입니다
        (마지막 열은 Jaw / 그리퍼이므로 제외해야 함).

    Returns
    -------
    raw_states : np.ndarray
        선택된 상태 열의 (N, D) 배열.
    action_label : str
        사람이 읽을 수 있는 액션 차원 설명.
    state_label : str
        사람이 읽을 수 있는 상태 차원 설명.
    sa_suffix : str
        출력 배열용 키 접미사 (제공됨, 변경 금지).
    
    """
    action_label, state_label, sa_suffix = _ACTION_SPACE_LABELS[action_space]

    if action_space == "ee":
        raw_states = merged["state_ee"][:, :3]
    elif action_space == "ee_full":
        raw_states = merged["state_ee"]
    elif action_space == "joints":
        raw_states = merged["state_joints"][:, :5]
    else:
        raise ValueError(f"Unknown action space: {action_space!r}")

    return raw_states, action_label, state_label, sa_suffix


def get_episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    """각 에피소드의 (시작, 종료) 인덱스 쌍을 반환합니다."""
    starts = np.concatenate([[0], episode_ends[:-1]])
    return list(zip(starts.tolist(), episode_ends.tolist()))


def compute_actions_for_episodes(
    states: np.ndarray,
    episode_ranges: list[tuple[int, int]],
    action_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """변화량 액션을 계산하고 정렬된 상태, 액션, episode_ends, 유지할 인덱스를 반환합니다.

    길이가 L인 각 에피소드에 대해 L-1개의 전이(transition)를 유지합니다:
        state[t] -> action = action_fn(state[t], state[t+1])   (t 범위: [start, end-2])
    기본값 (action_fn=None)은 단순 차를 사용합니다: state[t+1] - state[t].

    Returns:
        out_states:       (N', D) - 각 에피소드의 마지막 단계가 제거된 상태 배열
        out_actions:      (N', D) - 이에 대응하는 변화량 액션 배열
        out_episode_ends: (num_episodes,) - 잘라낸 데이터의 누적 종료 인덱스
        keep_idx:         (N',) - 유지된 타임스텝의 원본 인덱스 (다른 배열 정렬용)
    
    """
    out_states_list: list[np.ndarray] = []
    out_actions_list: list[np.ndarray] = []
    out_episode_ends: list[int] = []
    keep_idx_parts: list[np.ndarray] = []
    running = 0

    for start, end in episode_ranges:
        ep_states = states[start:end]  # (L, D)
        if ep_states.shape[0] < 2:
            continue  # 비정상(degenerate) 에피소드 제외
        out_states_list.append(ep_states[:-1])
        if action_fn is not None:
            out_actions_list.append(action_fn(ep_states[:-1], ep_states[1:]))
        else:
            out_actions_list.append(ep_states[1:] - ep_states[:-1])
        keep_idx_parts.append(np.arange(start, end - 1))
        running += ep_states.shape[0] - 1
        out_episode_ends.append(running)

    out_states = np.concatenate(out_states_list, axis=0)
    out_actions = np.concatenate(out_actions_list, axis=0)
    keep_idx = np.concatenate(keep_idx_parts)
    return out_states, out_actions, np.array(out_episode_ends, dtype=np.int64), keep_idx


def trim_to_transitions(
    merged: dict[str, np.ndarray],
    keep_idx: np.ndarray,
    *,
    skip_keys: set[str],
) -> dict[str, np.ndarray]:
    """보조 배열들을 유지된 전이 인덱스에 맞게 잘라냅니다.

    *merged*의 모든 배열(*skip_keys* 및 ``"episode_ends"`` 제외)에 ``keep_idx``를 적용하고,
    ``state_ee``의 이름을 ``state_ee_full``로 변경합니다 (ee-xyz 상태 키와의 충돌 방지).

    Parameters
    ----------
    merged : dict
        :func:`load_and_merge_zarrs`에서 얻은 원본 병합 데이터.
    keep_idx : (N',) array
        유지할 타임스텝의 인덱스 (:func:`compute_actions_for_episodes`에 의해 반환됨).
    skip_keys : set[str]
        이미 기록되어 건너뛰어야 할 키 세트.

    Returns
    -------
    dict[str, np.ndarray]
        최종 대상 이름으로 변경되고 잘라내진 배열들.
    
    """
    trimmed: dict[str, np.ndarray] = {}

    for key, arr in merged.items():
        if key == "episode_ends":
            continue
        if key.startswith("_"):
            continue  # 내부 메타데이터 제외 (예: _num_dagger_episodes)

        dest_name = key
        if dest_name == "state_ee":
            dest_name = "state_ee_full"  # state_ee_xyz와의 충돌 방지
        elif dest_name in ("pos_cube_red", "pos_cube_green", "pos_cube_blue"):
            dest_name = f"original_{dest_name}"

        if dest_name in skip_keys:
            continue

        sliced = arr[keep_idx]

        trimmed[dest_name] = sliced
    return trimmed


def load_and_merge_zarrs(zarr_paths: list[Path]) -> dict[str, np.ndarray]:
    """여러 zarr 저장소의 데이터를 로드하고 병합합니다.

    다음 키를 포함하는 딕셔너리를 반환합니다:
        state_joints, state_ee, state_cube, episode_ends,
        그리고 발견된 각 카메라에 대한 images_<cam>.
    
    """
    all_data: dict[str, list[np.ndarray]] = {}
    cumulative_offset = 0

    for zpath in sorted(zarr_paths):
        root = zarr.open_group(str(zpath), mode="r")
        data_grp = root["data"]
        meta_grp = root["meta"]

        ep_ends = np.asarray(meta_grp["episode_ends"])
        if ep_ends.size == 0:
            print(f"  Skipping {zpath} (no episodes)")
            continue

        n_steps = int(ep_ends[-1])
        is_dagger = "dagger" in str(zpath).lower()
        tag = " [dagger]" if is_dagger else ""
        print(f"  {zpath.name}: {ep_ends.size} episode(s), {n_steps} steps{tag}")

        # 누적 오프셋만큼 episode_ends 이동
        all_data.setdefault("episode_ends", []).append(ep_ends + cumulative_offset)
        all_data.setdefault("_dagger_ep_counts", []).append(
            ep_ends.size if is_dagger else 0
        )

        for key in data_grp:
            arr = np.asarray(data_grp[key][:n_steps])
            all_data.setdefault(key, []).append(arr)

        cumulative_offset += n_steps

    merged: dict[str, np.ndarray] = {}
    for key, arrays in all_data.items():
        if key == "_dagger_ep_counts":
            continue  # 배열이 아니므로 별도로 처리
        merged[key] = np.concatenate(arrays, axis=0)

    # dagger 에피소드 개수를 일반 int로 전달 (ndarray 아님)
    merged["_num_dagger_episodes"] = sum(all_data.get("_dagger_ep_counts", [0]))

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute actions from recorded teleop zarr datasets."
    )
    parser.add_argument(
        "--action-space",
        choices=list(_ACTION_SPACE_LABELS),
        required=True,
        help="Action space: ee (xyz, 3D), ee_full (pos+euler, 6D), joints (5D).",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("./datasets/raw/single_cube"),
        help="Root directory to search for .zarr stores (default: ./datasets/raw/single_cube).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output zarr path. Default: datasets/processed/<task>/processed_<action-space>.zarr",
    )
    args = parser.parse_args()

    # ── zarr 저장소 탐색 ──────────────────────────────────────────
    zarr_paths = sorted(args.datasets_dir.rglob("*.zarr"))
    if not zarr_paths:
        print(f"No .zarr stores found under {args.datasets_dir}")
        return
    print(f"Found {len(zarr_paths)} zarr store(s):")

    # ── 로드 및 병합 ──────────────────────────────────────────────────
    merged = load_and_merge_zarrs(zarr_paths)
    episode_ends = merged["episode_ends"]
    episode_ranges = get_episode_ranges(episode_ends)
    n_episodes = len(episode_ranges)
    n_dagger_episodes = int(merged.get("_num_dagger_episodes", 0))
    n_total = int(episode_ends[-1])
    print(
        f"\nMerged: {n_episodes} episodes ({n_dagger_episodes} dagger), {n_total} total steps"
    )

    # ── 선택한 액션 공간에 대한 상태 배열 선택 ────────────────
    raw_states, action_label, state_label, sa_suffix = select_action_space(
        args.action_space, merged
    )
    print(
        f"Action space: {args.action_space}, state_dim={raw_states.shape[1]} ({state_label}), action=({action_label})"
    )

    # ── 다음 상태 기반 액션 계산 ────────────────────────────────────
    action_fn = _ee_full_delta if args.action_space == "ee_full" else None
    states, actions, new_ep_ends, keep_idx = compute_actions_for_episodes(
        raw_states,
        episode_ranges,
        action_fn=action_fn,
    )
    print(
        f"After action computation: {states.shape[0]} transitions "
        f"across {new_ep_ends.size} episodes"
    )

    # ── 그리퍼 액션 정렬 (계산값이 아닌 기록된 제어 명령) ───────────
    raw_action_gripper = merged.get("action_gripper")
    if raw_action_gripper is not None:
        action_gripper_trimmed = raw_action_gripper[keep_idx]
    # ── 출력 zarr 쓰기 ─────────────────────────────────────────────
    if args.output is not None:
        out_path = args.output
    else:
        # 기본값: datasets/processed/<task>/processed_<action-space>.zarr
        base_dir = Path("./datasets/processed")
        if "multi_cube" in str(args.datasets_dir):
            base_dir = base_dir / "multi_cube"
        else:
            base_dir = base_dir / "single_cube"
        out_path = base_dir / f"processed_{sa_suffix}.zarr"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting to {out_path} ...")

    out_root = zarr.open_group(str(out_path), mode="w", zarr_format=3)
    compressor = zarr.codecs.Blosc(cname="zstd", clevel=3, shuffle=2)
    compressors = (compressor,)

    out_data = out_root.require_group("data")
    out_meta = out_root.require_group("meta")

    state_key = f"state_{sa_suffix}"
    action_key = f"action_{sa_suffix}"

    # ee/관절에 대한 상태(state) 및 액션(action)
    out_data.create_array(
        state_key, data=states.astype(np.float32), compressors=compressors
    )
    out_data.create_array(
        action_key, data=actions.astype(np.float32), compressors=compressors
    )

    # 그리퍼 액션 (동일한 타임스텝에 정렬되어 기록된 제어 명령)
    out_data.create_array(
        "action_gripper",
        data=action_gripper_trimmed.astype(np.float32),
        compressors=compressors,
    )

    # 에피소드 종료 지점 (episode_ends)
    out_meta.create_array(
        "episode_ends", data=new_ep_ends.astype(np.int64), compressors=compressors
    )

    # 보조 배열 잘라내기 및 복사 (이미지, 큐브 상태, 원본 상태, 그리퍼 상태)
    already_written = {state_key, action_key, "action_gripper"}
    aux_arrays = trim_to_transitions(merged, keep_idx, skip_keys=already_written)
    for dest_name, data in aux_arrays.items():
        out_data.create_array(dest_name, data=data, compressors=compressors)

    # 메타데이터
    out_root.attrs["action_space"] = args.action_space
    out_root.attrs["action_dim"] = int(actions.shape[1])
    out_root.attrs["action_spec"] = action_label
    out_root.attrs["state_spec"] = state_label
    out_root.attrs["state_key"] = state_key
    out_root.attrs["action_key"] = action_key
    out_root.attrs["num_episodes"] = int(new_ep_ends.size)
    out_root.attrs["num_dagger_episodes"] = n_dagger_episodes
    out_root.attrs["num_transitions"] = int(states.shape[0])
    out_root.attrs["source_zarrs"] = [str(p) for p in zarr_paths]

    print(f"Done. {states.shape[0]} transitions written.")
    print(f"  data/{state_key}:  {states.shape}")
    print(f"  data/{action_key}: {actions.shape}")
    print(f"  data/action_gripper: {action_gripper_trimmed.shape}")
    print(f"  meta/episode_ends: {new_ep_ends.shape}")


if __name__ == "__main__":
    main()
