"""DAgger 대화형 평가: 인간 개입(takeover) 기능과 함께 학습된 정책을 실행합니다.

장애물 씬에서 정책 추론을 실행합니다. 언제든지 전문가(사용자)가 개입 키를 눌러 수동 제어로 전환할 수 있습니다.
개입 모드에 있는 동안 모든 타임스텝은 zarr 저장소에 기록됩니다(record_teleop_demos.py와 동일한 형식).
개입 키를 다시 누르거나 에피소드가 종료되면 제어권이 다시 정책으로 넘어갑니다.

수집된 데이터는 datasets/raw/single_cube/dagger/ 아래에 저장되며,
이후 compute_actions.py를 통해 기존 데모 데이터와 병합하여 재학습에 사용할 수 있습니다. 이는 학습을 위한 병합된 .zarr 데이터셋을 생성합니다.

사용법:
    python scripts/dagger_eval.py \\
        --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt \\
        --num-episodes 10
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from hw3.dataset import Normalizer
from hw3.eval_utils import (
    apply_action,
    check_cube_out_of_bounds,
    check_success,
    infer_action_chunk,
    load_checkpoint,
)
from hw3.sim_env import (
    SO100SimEnv,
)
from hw3.teleop_utils import (
    CAMERA_NAMES,
    DEFAULT_KEYMAP_PATH,
    ZarrEpisodeWriter,
    compose_camera_views,
    handle_teleop_key,
    load_keymap,
)
from so101_gym.constants import ASSETS_DIR

XML_PATH = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"


# ── 메인 DAgger 루프 ─────────────────────────────────────────────────


def run_dagger_episode(
    env: SO100SimEnv,
    model: torch.nn.Module,
    normalizer: Normalizer,
    state_keys: list[str],
    action_keys: list[str],
    device: torch.device,
    writer: ZarrEpisodeWriter,
    key_to_action: dict[int, str],
    *,
    max_steps: int = 800,
    successes: int = 0,
    total: int = 0,
    headless: bool = False,
) -> tuple[bool, int, bool, bool]:
    """하나의 DAgger 에피소드를 실행합니다: 정책이 실행되며, 인간이 언제든지 개입할 수 있습니다.

    반환값: (success, n_takeover_steps, aborted, replay).
    
    """
    # 호출자가 리플레이를 위해 복원할 수 있도록 RNG 상태를 저장
    rng_state_before_reset = env.rng.bit_generator.state
    obs = env.reset()

    action_queue: list[np.ndarray] = []
    step = 0
    success = False
    human_control = False
    n_takeover_steps = 0
    recording_this_episode = False  # 기록된 데이터가 있는지 추적
    # 유예 기간: 인간 제어 중 성공이 감지된 후, 최소 하나의 전체 청크가
    # 기록되도록 GRACE_SECS 동안 계속 실행합니다.
    GRACE_SECS = 1.7
    grace_steps_remaining: int | None = None  # None = 유예 기간이 아님

    while step < max_steps or human_control:
        # ── 키보드 처리 (헤드리스 모드에서는 건너뜀) ─────────────
        if not headless:
            k_raw = cv2.waitKeyEx(1)
        else:
            k_raw = -1
        if k_raw != -1:
            action_name = key_to_action.get(k_raw)

            if action_name == "escape":
                # 현재 에피소드 데이터를 폐기하고 중단
                if recording_this_episode:
                    writer.discard_episode()
                    print("  Episode discarded on escape.")
                return success, n_takeover_steps, True, False  # aborted

            if action_name == "record":
                # 인간 개입(takeover) 모드 토글
                human_control = not human_control
                if human_control:
                    print("  >>> HUMAN TAKEOVER — you are now controlling the arm")
                    print("      Press your 'record' key again to hand back to policy")
                    action_queue.clear()  # 대기열에 있는 모든 정책 액션 제거
                    recording_this_episode = True
                else:
                    print("  <<< POLICY RESUMED")

            if action_name == "reset":
                # 리플레이: 데이터를 폐기하고 동일한 무작위성(randomization)으로 반복
                if recording_this_episode:
                    writer.discard_episode()
                    print("  Episode discarded — replaying same scenario.")
                # 다음 reset()이 동일한 에피소드를 재현하도록 RNG 복원
                env.rng.bit_generator.state = rng_state_before_reset
                return False, 0, False, True  # replay=True

            # Enter 키 (13 / 0x0D) = 다음 에피소드로 건너뛰기
            if k_raw == 13 or k_raw == 0x0D:
                if recording_this_episode:
                    writer.discard_episode()
                    print("  Episode discarded — skipping to next.")
                return False, 0, False, False  # replay=False

            # 인간 제어 상태인 경우, 이동 키 적용
            if human_control and action_name is not None:
                handle_teleop_key(
                    action_name,
                    env.data,
                    env.model,
                    env.mocap_id,
                    env.act_ids[env._jaw_idx],
                )

        # ── step 이전 상태 기록 (인간이 제어 중인 경우) ────────
        if human_control:
            # DAgger를 위한 현재 상태 기록
            joints = env.get_joint_angles()
            ee_state = env.get_ee_state()
            cube_state = env.get_cube_state()
            gripper_state = np.array([env.get_gripper_angle()], dtype=np.float32)
            action_gripper = np.array(
                [env.data.ctrl[env.act_ids[env._jaw_idx]]], dtype=np.float32
            )
            obstacle_state = env.get_obstacle_pos()
            writer.append(
                joints,
                ee_state,
                cube_state,
                gripper_state,
                action_gripper,
                obstacle_state,
            )
            n_takeover_steps += 1

        # ── 정책 추론 (인간 제어 상태가 아닌 경우) ────────────────
        if not human_control:
            if not action_queue:
                chunk = infer_action_chunk(
                    model=model,
                    normalizer=normalizer,
                    obs=obs,
                    state_keys=state_keys,
                    device=device,
                )
                action_queue.extend(chunk)

            action = action_queue.pop(0)
            apply_action(env, action, action_keys)

        # ── 시뮬레이션 step 실행 ───────────────────────────────────────────
        obs = env.step()
        step += 1

        # ── 종료 조건 확인 ─────────────────────────────────────────
        success = check_success(env)
        if success:
            if human_control and grace_steps_remaining is None:
                # 기록을 계속 유지하기 위해 유예 기간 시작
                grace_steps_remaining = int(GRACE_SECS / env.dt_ctrl)
                print(f"  Cube in bin! Recording {grace_steps_remaining} more "
                      f"steps ({GRACE_SECS}s grace period)...")
            elif not human_control:
                # 정책 모드 — 즉시 종료
                if recording_this_episode:
                    writer.end_episode()
                    print(f"  DAgger episode saved ({n_takeover_steps} takeover steps)")
                return success, n_takeover_steps, False, False

        # 유예 기간 카운트다운
        if grace_steps_remaining is not None:
            grace_steps_remaining -= 1
            if grace_steps_remaining <= 0:
                if recording_this_episode:
                    writer.end_episode()
                    print(f"  DAgger episode saved ({n_takeover_steps} takeover steps)")
                return True, n_takeover_steps, False, False

        if check_cube_out_of_bounds(env):
            print("  Cube out of bounds — early termination.")
            if recording_this_episode:
                writer.end_episode()
                print(f"  DAgger episode saved ({n_takeover_steps} takeover steps)")
            return False, n_takeover_steps, False, False

        # ── 렌더링 (헤드리스 모드에서는 건너뜀) ────────────────────────
        if headless:
            continue

        img = compose_camera_views({cam: env.render(cam) for cam in CAMERA_NAMES})
        status = f"Step {step}/{max_steps}"
        if human_control:
            status += " | HUMAN CONTROL"
        else:
            status += f" | POLICY (queue {len(action_queue)})"
        cv2.putText(
            img, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )

        # 성공률
        if total > 0:
            rate = successes / total * 100
            sr_text = f"Success: {successes}/{total} ({rate:.0f}%)"
        else:
            sr_text = "Success: -/-"
        color = (0, 255, 0) if success else (0, 0, 255)
        cv2.putText(img, sr_text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # DAgger 정보
        dagger_text = (
            f"DAgger steps: {n_takeover_steps} | Episodes saved: {writer.num_episodes}"
        )
        cv2.putText(
            img, dagger_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2
        )

        # 모드 표시기
        if human_control:
            cv2.putText(
                img, "HUMAN", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
            )
        else:
            cv2.putText(
                img, "POLICY", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
            )

        # 힌트
        def _label_for(act):
            for code, a in key_to_action.items():
                if a == act:
                    if 32 <= (code & 0xFF) <= 126:
                        ch = chr(code & 0xFF)
                        return ch if ch.strip() else "SPACE"
                    if code & 0xFF == 27:
                        return "ESC"
                    return f"key:{code}"
            return "?"

        hint = (
            f"{_label_for('record')} takeover | "
            f"{_label_for('reset')} replay | "
            f"ENTER skip | "
            f"{_label_for('escape')} quit"
        )
        cv2.putText(
            img,
            hint,
            (10, img.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.imshow("DAgger Eval", img)
        time.sleep(env.dt_ctrl)

    # max_steps에 도달하여 에피소드 종료됨
    if recording_this_episode:
        writer.end_episode()
        print(f"  DAgger episode saved ({n_takeover_steps} takeover steps)")
    return success, n_takeover_steps, False, False


def main():
    parser = argparse.ArgumentParser(
        description="DAgger interactive evaluation with human takeover."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the model checkpoint (.pt).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible cube spawns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for DAgger zarr data (default: datasets/raw/single_cube/dagger/<timestamp>).",
    )
    parser.add_argument(
        "--keymap",
        type=Path,
        default=None,
        help="Path to keymap.json (default: hw3/keymap.json).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without rendering or real-time pacing (faster batch eval). "
        "No human takeover is possible in this mode.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 모델 로드
    model, normalizer, chunk_size, state_keys, action_keys = load_checkpoint(
        args.checkpoint, device
    )

    # use_mocap 사용 여부 결정
    use_mocap = not any("action_joints" in k for k in action_keys)

    # 씬(Scene)
    print(f"Scene: {XML_PATH.name}")

    env = SO100SimEnv(
        xml_path=XML_PATH,
        render_w=640,
        render_h=480,
        use_mocap=use_mocap,
        obstacle_mode="adversarial",
        seed=args.seed,
    )

    # 키맵
    km_path = args.keymap or DEFAULT_KEYMAP_PATH
    key_to_action = load_keymap(km_path)
    print(f"Loaded keymap from {km_path}")

    # DAgger 출력 zarr
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path("./datasets/raw/single_cube/dagger") / ts
    out_zarr = out_dir / "so100_transfer_cube_teleop.zarr"
    print(f"DAgger data will be saved to: {out_zarr}")

    writer = ZarrEpisodeWriter(
        path=out_zarr,
    )

    if not args.headless:
        cv2.namedWindow("DAgger Eval", cv2.WINDOW_AUTOSIZE)

    successes = 0
    total_takeover_steps = 0
    try:
        ep = 0
        while ep < args.num_episodes:
            ep += 1
            print(f"\n═══ DAgger Episode {ep}/{args.num_episodes} ═══")
            print("  Policy is running. Press your 'record' key to take over control.")

            success, n_takeover, aborted, replay = run_dagger_episode(
                env,
                model,
                normalizer,
                state_keys,
                action_keys,
                device,
                writer,
                key_to_action,
                max_steps=args.max_steps,
                successes=successes,
                total=ep - 1,
                headless=args.headless,
            )

            if aborted:
                print("Aborted by user.")
                break

            if replay:
                # run_dagger_episode 내부에서 RNG가 이미 복원됨
                print("  Replaying same episode...")
                ep -= 1  # 이번 시도는 횟수에 포함하지 않음
                continue

            total_takeover_steps += n_takeover
            if success:
                successes += 1
            rate = successes / ep * 100
            result = "SUCCESS" if success else "FAIL"
            print(f"Episode {ep}: {result} | takeover steps this ep: {n_takeover}")
            print(f"  Success rate: {successes}/{ep} ({rate:.0f}%)")

    finally:
        writer.flush()
        cv2.destroyAllWindows()

    n_eps = writer.num_episodes
    n_steps = writer.num_steps_total
    rate = successes / max(1, args.num_episodes) * 100
    print(f"\n{'=' * 50}")
    print("DAgger session complete.")
    print(f"  Episodes evaluated: {args.num_episodes}")
    print(f"  Success rate: {successes}/{args.num_episodes} ({rate:.0f}%)")
    print(f"  Total takeover steps: {total_takeover_steps}")
    print(f"  DAgger episodes saved: {n_eps} ({n_steps} total steps)")
    print(f"  Data saved to: {out_zarr}")
    print("\n If you collected data, you can now retrain your model with the additional episodes.")


if __name__ == "__main__":
    main()
