import numpy as np

from envs.rotation_utils import (
    quat_mul,
    quat_conjugate,
    quat_normalize,
    rot_mat_to_quat,
)

def reset_robot(default_qpos: np.ndarray) -> np.ndarray:
    """
    각 관절에 대해 [-0.5, 0.5] 범위의 작은 균등 노이즈를 추가하여 기본 설정 주변으로 로봇 관절 위치를 재설정합니다.
    
    """
    default_qpos = np.asarray(default_qpos, dtype=np.float64)
    noise = np.random.uniform(-0.5, 0.5, size=default_qpos.shape)
    return default_qpos + noise


def reset_target_position(base_pos: np.ndarray) -> np.ndarray:
    """
    로봇 베이스 주변의 무작위 목표 위치를 샘플링합니다.

    목표 위치는 베이스를 기준으로 다음 범위에서 샘플링됩니다:
        x in [0.2, 0.4]
        y in [-0.2, 0.2]
        z in [0.1, 0.4]
    
    """
    base_pos = np.asarray(base_pos, dtype=np.float64)
    offset = np.random.uniform(
        low=np.array([0.2, -0.2, 0.1], dtype=np.float64),
        high=np.array([0.4, 0.2, 0.4], dtype=np.float64),
    )
    return base_pos + offset


def process_action(action: np.ndarray, jnt_range: np.ndarray) -> np.ndarray:
    """
    [-1, 1] 범위의 정규화된 행동(action)을 물리적 관절 범위로 선형 매핑합니다.
    
    """
    action = np.asarray(action, dtype=np.float64)
    jnt_range = np.asarray(jnt_range, dtype=np.float64)

    action = np.clip(action, -1.0, 1.0)
    low = jnt_range[:, 0]
    high = jnt_range[:, 1]
    target_qpos = (action + 1.0) * 0.5 * (high - low) + low
    return target_qpos


def compute_reward(ee_tracking_error: float, q_vel: np.ndarray) -> float:
    """
    말단 장치(end-effector) 위치 추종에 대한 보상.

    dense_reward = exp(-2 * error)
    sparse_reward = 1.0 if error < 0.005 else 0.0
    total_reward = dense_reward + sparse_reward
    
    """
    reward = np.exp(-10.0 * ee_tracking_error)
    if ee_tracking_error < 0.10:
        reward += 0.2
    if ee_tracking_error < 0.05:
        reward += 0.2
    if ee_tracking_error < 0.02:
        reward += 0.5
    if ee_tracking_error < 0.005:
        reward += 0.5
    reward -= 0.01 * np.max(np.square(q_vel))  # 관절 속도에 대한 작은 페널티
    return float(reward)


def get_obs(
    qpos: np.ndarray,
    ee_pos_w: np.ndarray,
    ee_rot_w: np.ndarray,
    base_pos_w: np.ndarray,
    base_rot_w: np.ndarray,
    target_pos_w: np.ndarray,
) -> np.ndarray:
    """
    로봇 베이스 좌표계에서 관측(observation) 벡터를 구성합니다.

    관측 레이아웃:
        [
            qpos,
            ee_pos_base,
            ee_quat_base,
            target_pos_base,
        ]
    
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    ee_pos_w = np.asarray(ee_pos_w, dtype=np.float64)
    ee_rot_w = np.asarray(ee_rot_w, dtype=np.float64)
    base_pos_w = np.asarray(base_pos_w, dtype=np.float64)
    base_rot_w = np.asarray(base_rot_w, dtype=np.float64)
    target_pos_w = np.asarray(target_pos_w, dtype=np.float64)

    ee_pos_base = base_rot_w.T @ (ee_pos_w - base_pos_w)

    base_quat_w = rot_mat_to_quat(base_rot_w)
    ee_quat_w = rot_mat_to_quat(ee_rot_w)
    ee_quat_base = quat_mul(quat_conjugate(base_quat_w), ee_quat_w)
    ee_quat_base = quat_normalize(ee_quat_base)

    target_pos_base = base_rot_w.T @ (target_pos_w - base_pos_w)
    pos_error_base = target_pos_base - ee_pos_base

    obs = np.concatenate([qpos, ee_pos_base, ee_quat_base, target_pos_base, pos_error_base])
    return obs.astype(np.float32)