"""정책 평가(``eval.py``, ``dagger_eval.py``)를 위한 공통 유틸리티.

제공 기능:
- ``ZARR_KEY_TO_OBS`` — zarr 배열 이름에서 sim 관측(observation) 추출기로의 매핑.
- ``parse_key_spec`` — ``"key[:3]"``을 ``("key", slice(None, 3))``으로 분할.
- ``load_checkpoint`` — ``.pt`` 파일에서 모델, 정규화기(normalizer) 및 메타데이터 로드.
- ``obs_to_state`` — sim obs 딕셔너리로부터 평탄화된 state 벡터 조립.
- ``action_key_dim`` — 각 action key 기본 이름의 차원.
- ``apply_action`` — 시뮬레이션에 단일 예측 delta action 적용.
- ``check_success`` — 큐브가 빈(bin) 안에 있는지 여부 확인.
- ``check_cube_out_of_bounds`` — 큐브가 작업 공간(workspace)을 벗어났는지 여부 확인.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hw3.dataset import Normalizer
from hw3.model import build_policy
from hw3.sim_env import CUBE_COLORS, SO100MulticubeSimEnv, SO100SimEnv

# ── quaternion / euler 헬퍼 함수 (wxyz 컨벤션) ──────────────────────────────


def _euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """Euler 각도(roll, pitch, yaw)를 wxyz 단위 quaternion으로 변환."""
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=euler.dtype)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """두 wxyz quaternion의 Hamilton 곱."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=q1.dtype)

# ── zarr key → obs 매핑 ───────────────────────────────────────────

ZARR_KEY_TO_OBS: dict[str, callable] = {
    "state_ee_xyz": lambda obs: obs["ee_pos"],
    "state_ee_full": lambda obs: obs["ee"],
    "state_joints": lambda obs: obs["joints"],
    "state_gripper": lambda obs: obs["gripper"],
    "action_gripper": lambda obs: obs["gripper"],
    "state_cube": lambda obs: obs["cube"],
    "state_obstacle": lambda obs: obs["obstacle"],
    "goal_pos": lambda obs: obs["goal_pos"],
    # Multicube / goal-conditioned
    "original_pos_cube_red": lambda obs: obs["cubes"][:7],
    "original_pos_cube_green": lambda obs: obs["cubes"][7:14],
    "original_pos_cube_blue": lambda obs: obs["cubes"][14:21],
    "state_goal": lambda obs: obs["goal"],
}


# ── key spec 파싱 ─────────────────────────────────────────────────


def parse_key_spec(spec: str) -> tuple[str, slice]:
    """``"key[:3]"``을 파싱하여 ``("key", slice(None, 3))``으로 반환."""
    if "[" not in spec:
        return spec, slice(None)
    name, rest = spec.split("[", 1)
    rest = rest.rstrip("]")
    parts = rest.split(":")
    if len(parts) == 2:
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        return name, slice(start, stop)
    raise ValueError(f"Invalid key spec: {spec!r}")


# ── 체크포인트 로딩 ────────────────────────────────────────────────


def load_checkpoint(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Normalizer, int, list[str], list[str]]:
    """모델, 정규화기(normalizer), chunk_size, state_keys, action_keys를 로드합니다.

    필요한 모든 메타데이터는 체크포인트 자체에서 읽어옵니다.

    ``(model, normalizer, chunk_size, state_keys, action_keys)``를 반환합니다.
    
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    chunk_size = int(ckpt["chunk_size"])
    state_keys: list[str] = ckpt["state_keys"]
    action_keys: list[str] | None = ckpt.get("action_keys")

    norm_data = ckpt["normalizer"]
    normalizer = Normalizer(
        state_mean=np.asarray(norm_data["state_mean"], dtype=np.float32),
        state_std=np.asarray(norm_data["state_std"], dtype=np.float32),
        action_mean=np.asarray(norm_data["action_mean"], dtype=np.float32),
        action_std=np.asarray(norm_data["action_std"], dtype=np.float32),
    )

    d_model = int(ckpt.get("d_model", 128))
    depth = int(ckpt.get("depth", 2))
    policy_type = str(ckpt.get("policy_type", "obstacle"))
    model = build_policy(
        policy_type,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        d_model=d_model,
        depth=depth,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from {ckpt_path}")
    print(
        f"  policy_type={policy_type}, epoch={ckpt.get('epoch', '?')}, "
        f"val_loss={ckpt.get('val_loss', 0):.6f}"
    )
    print(f"  state_keys={state_keys}, action_keys={action_keys}")
    print(f"  state_dim={state_dim}, action_dim={action_dim}, chunk_size={chunk_size}")

    return model, normalizer, chunk_size, state_keys, action_keys


# ── state 조립 ────────────────────────────────────────────────────


def obs_to_state(obs: dict[str, np.ndarray], state_keys: list[str]) -> np.ndarray:
    """학습 key spec을 사용하여 sim obs 딕셔너리로부터 평탄화된 state 벡터를 조립합니다."""
    parts: list[np.ndarray] = []
    for spec in state_keys:
        name, col_slice = parse_key_spec(spec)
        if name not in ZARR_KEY_TO_OBS:
            raise ValueError(
                f"Unknown state key '{name}'. Known keys: {list(ZARR_KEY_TO_OBS)}"
            )
        raw = ZARR_KEY_TO_OBS[name](obs)
        if name == "state_joints":
            raw = raw[:5]
        parts.append(raw[col_slice] if col_slice != slice(None) else raw)
    return np.concatenate(parts).astype(np.float32)


# ── 추론(inference) 헬퍼 함수 ────────────────────────────────────────────────


def infer_action_chunk(
    model: torch.nn.Module,
    normalizer: Normalizer,
    obs: dict[str, np.ndarray],
    state_keys: list[str],
    device: torch.device,
) -> np.ndarray:
    """정책의 순방향 패스(forward pass)를 한 번 실행하고 역정규화된 action chunk를 반환합니다."""
    state = obs_to_state(obs, state_keys)
    state_norm = normalizer.normalize_state(state)
    state_t = torch.from_numpy(state_norm).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model.sample_actions(state_t)

    chunk = pred.squeeze(0).cpu().numpy()
    for i in range(chunk.shape[0]):
        chunk[i] = normalizer.denormalize_action(chunk[i])
    return chunk


# ── action 헬퍼 함수 ────────────────────────────────────────────────────


def action_key_dim(key_name: str) -> int:
    """주어진 action key 기본 이름에 대해 기대되는 raw 차원을 반환합니다."""
    dims = {
        "action_ee_xyz": 3,
        "action_ee_full": 6,
        "action_gripper": 1,
        "action_joints": 5,
    }
    return dims.get(key_name, 0)


def apply_action(env: SO100SimEnv, action: np.ndarray, action_keys: list[str]) -> None:
    """시뮬레이션에 단일 예측 delta action을 적용합니다.

    모델은 ee/joint action에 대해 **상대적인 변화량**(delta)을 예측합니다.
    이 함수는 예측된 벡터를 다시 key별 세그먼트로 분할하고,
    시뮬레이션에서 현재 state를 읽어 delta를 더한 뒤,
    그 결과로 나온 절대 목표치(absolute target)를 적용합니다.

    그리퍼(gripper) action은 있는 그대로 적용됩니다(delta가 아닌 절대 제어 명령).
    
    """
    offset = 0
    for spec in action_keys:
        name, col_slice = parse_key_spec(spec)
        full_dim = action_key_dim(name)
        if full_dim == 0:
            raise ValueError(f"Unknown action key: {name}")

        idx = np.arange(full_dim)[col_slice]
        seg_dim = len(idx)
        segment = action[offset : offset + seg_dim]
        offset += seg_dim

        if seg_dim < full_dim:
            full_vec = np.zeros(full_dim, dtype=action.dtype)
            full_vec[idx] = segment
        else:
            full_vec = segment

        if name == "action_ee_xyz":
            current_target = env.data.mocap_pos[env.mocap_id].copy()
            env.set_mocap_pos(current_target + full_vec[:3])
        elif name == "action_ee_full":
            current_pos_target = env.data.mocap_pos[env.mocap_id].copy()
            current_quat_target = env.data.mocap_quat[env.mocap_id].copy()
            env.set_mocap_pos(current_pos_target + full_vec[:3])
            # euler delta → quaternion 변환 후 현재 orientation과 합성
            delta_quat = _euler_to_quat(full_vec[3:6])
            new_quat = _quat_multiply(delta_quat, current_quat_target)
            new_quat /= np.linalg.norm(new_quat)  # 정규화 재수행
            env.set_mocap_quat(new_quat)
        elif name == "action_gripper":
            env.set_gripper(float(full_vec[0]))
        elif name == "action_joints":
            current_targets = np.array(
                [env.data.ctrl[env.act_ids[i]] for i in range(5)],
                dtype=action.dtype,
            )
            new_targets = current_targets + full_vec
            env.set_targets(new_targets)


# ── 성공 / 경계 벗어남(bounds) 검사 ─────────────────────────────────────────


def check_success(
    env: SO100SimEnv, xy_thresh: float = 0.05, z_thresh: float = 0.04
) -> bool:
    """큐브가 빈(bin) 안에 있는지 확인합니다.

    시뮬레이션의 현재 빈 중심(bin centre)을 사용합니다.
    
    """
    cube = env.get_cube_state()[:3]
    if isinstance(env, SO100MulticubeSimEnv):
        xy_thresh = min(xy_thresh, 0.04)
    bin_center = env.get_goal_pos()[:2]
    in_xy = np.all(np.abs(cube[:2] - bin_center) < xy_thresh)
    in_z = 0.0 < cube[2] < z_thresh
    return bool(in_xy and in_z)


def check_cube_out_of_bounds(
    env: SO100SimEnv,
    x_range: tuple[float, float] = (-0.6, 0.6),
    y_range: tuple[float, float] = (0.1, 1.1),
    z_min: float = -0.01,
) -> bool:
    """큐브가 테이블에서 떨어졌거나 작업 공간(workspace)을 벗어난 경우 True를 반환합니다."""
    cube = env.get_cube_state()[:3]
    if cube[2] < z_min:
        return True
    if cube[0] < x_range[0] or cube[0] > x_range[1]:
        return True
    if cube[1] < y_range[0] or cube[1] > y_range[1]:
        return True
    return False


def check_wrong_cube_in_bin(
    env: SO100MulticubeSimEnv,
    xy_thresh: float = 0.04,
    z_thresh: float = 0.04,
) -> str | None:
    """목표가 아닌(non-goal) 큐브가 빈(bin) 안에 있는지 확인합니다.

    빈 안에서 발견된 첫 번째 잘못된 큐브의 색상 이름을 반환하며,
    잘못된 큐브가 없으면 ``None``을 반환합니다.
    
    """
    bin_center = env.get_goal_pos()[:2]
    goal_idx = env._goal_index

    for i, color in enumerate(CUBE_COLORS):
        if i == goal_idx:
            continue
        cube_pos = env.data.qpos[env.cube_qpos_slices[i][:3]].copy()
        in_xy = np.all(np.abs(cube_pos[:2] - bin_center) < xy_thresh)
        in_z = 0.0 < cube_pos[2] < z_thresh
        if in_xy and in_z:
            return color
    return None
