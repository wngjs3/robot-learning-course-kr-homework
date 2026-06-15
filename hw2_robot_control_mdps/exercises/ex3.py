import numpy as np

import __init__
from scripts.utils import quat_mul, quat_conjugate, quat_normalize, rot_mat_to_quat

"""
# Important note:
# In physical simulations in Python, it is necessary to correctly modify the values in arrays which are attributes
# of the data object. Be careful to modify the arrays in-place (e.g., using slicing array[:] = new_array) rather than overwriting the
# entire array reference, otherwise the physics engine will not see your changes!
"""

def reset_robot(default_qpos: np.ndarray) -> np.ndarray:
    """
    TODO: 약간의 균일 노이즈(-0.5, 0.5)를 추가하여 로봇을 기본 관절 위치로 리셋하는 기능을 구현하세요.
    np.random.uniform을 사용하여 기본 관절 위치에 무작위 노이즈를 추가할 수 있습니다.
    
    입력:
    - default_qpos: np.ndarray. 기본 관절 위치. 차원: 1D 배열, 형상: (num_joints,).

    반환:
    - reset_qpos: np.ndarray. 로봇을 리셋할 관절 위치. 차원: 1D 배열, 형상: (num_joints,).
    
    """
    raise NotImplementedError()
    


def reset_target_position(base_pos: np.ndarray) -> np.ndarray:
    """
    TODO: 균일 분포로부터 베이스 기준의 새로운 무작위 목표 위치를 샘플링하고 계산하세요.
    균일 분포의 범위는 다음 배열로 주어집니다:
    - x: [0.2, 0.4]
    - y: [-0.2, 0.2]
    - z: [0.1, 0.4]

    입력:
    - base_pos: np.ndarray. 로봇 베이스의 3D 위치. 차원: 1D 배열, 형상: (3,).
    
    반환:
    - target_pos: np.ndarray. 베이스 기준 목표물의 3D 위치. 차원: 1D 배열, 형상: (3,).
    
    """
    raise NotImplementedError()


def process_action(action: np.ndarray, jnt_range: np.ndarray) -> np.ndarray:
    """
    TODO: 정규화된 액션 [-1, 1]을 목표 관절 위치로 변환하세요.
    
    정규화된 액션 [-1, 1]을 jnt_range로 정의된 실제 관절 범위로 매핑해야 합니다. 매핑은 선형적이어야 하며,
    -1은 관절의 하한값, 1은 관절의 상한값,
    그리고 0은 관절 범위의 중간값에 대응해야 합니다.

    입력:
    - action: np.ndarray. 정책(policy)으로부터 얻은 정규화된 액션. 차원: 1D 배열, 형상: (num_joints,).
    - jnt_range: np.ndarray. 관절의 하한 및 상한값. 차원: 2D 배열, 형상: (num_joints, 2).

    반환:
    - target_qpos: np.ndarray. 제어로 적용할 목표 관절 위치. 차원: 1D 배열, 형상: (num_joints,).
    
    """
    raise NotImplementedError()


def compute_reward(ee_tracking_error: float) -> float:
    """
    TODO: 
    목표물까지의 거리(오차)를 기반으로 보상을 계산하세요.
    강의 슬라이드에서 배웠듯이 보상에는 다양한 유형(예: dense, sparse)이 있습니다.
    보상 설계 시 이러한 접근 방식을 결합하는 것이 유용한 경우가 많습니다.
    이번 과제에서는 큰 속도 및 가속도에 대한 패널티 부여와 같은 고급 보상 엔지니어링까지 고려하는 것은 요구하지 않습니다.
    보너스 문제를 위해 자신만의 보상 함수를 직접 설계해 볼 수 있습니다.

    보상 함수의 설명:
    - dense_reward = exp(-2 * ee_tracking_error)
    - sparse_reward = ee_tracking_error < 0.005 일 때 1.0, 그렇지 않으면 0.0
    - reward = dense_reward + sparse_reward

    입력:
    - ee_tracking_error: float. 엔드이펙터와 목표 지점 사이의 거리. 차원: 스칼라

    반환:
    - reward: float. 추적 오차를 기반으로 계산된 보상. 차원: 스칼라
    
    """
    raise NotImplementedError()


def get_obs(qpos: np.ndarray, ee_pos_w: np.ndarray, ee_rot_w: np.ndarray, base_pos_w: np.ndarray, base_rot_w: np.ndarray, target_pos_w: np.ndarray) -> np.ndarray:
    """
    TODO: 환경의 로봇 상태 변수로부터 관측(observation) 벡터를 추출하세요.

     Mujoco에서는 상태를 월드 프레임에서 직접 접근할 수 있습니다. 하지만 정책 일반화(policy generalization)를 위해서는
     상태를 월드 프레임 대신 로봇의 베이스 프레임으로 표현하는 것이 중요합니다. 그래야 정책이 월드 내 로봇의 절대적 위치에 무관하게 유지될 수 있습니다.
    
    입력:
    - qpos: np.ndarray. 현재 관절 위치. 차원: 1D 배열, 형상: (num_joints,).
    - ee_pos_w: np.ndarray. 월드 프레임 기준 현재 엔드이펙터의 3D 위치. 차원: 1D 배열, 형상: (3,).
    - ee_rot_w: np.ndarray. 월드 프레임 기준 현재 엔드이펙터의 3D 회전 행렬. 차원: 2D 배열, 형상: (3, 3).
    - base_pos_w: np.ndarray. 월드 프레임 기준 현재 베이스의 3D 위치. 차원: 1D 배열, 형상: (3,).
    - base_rot_w: np.ndarray. 월드 프레임 기준 현재 베이스의 3D 회전 행렬. 차원: 2D 배열, 형상: (3, 3).
    - target_pos_w: np.ndarray. 월드 프레임 기준 현재 목표물의 3D 위치. 차원: 1D 배열, 형상: (3,).

    반환:
    - obs: np.ndarray. 다음 로봇 상태 변수들을 순서대로 포함하는 관측 벡터:
        [
            - 관절 위치 (qpos)
            - 로봇 베이스 프레임 기준 엔드이펙터 위치 (ee_pos_base)
            - 로봇 베이스 프레임 기준 엔드이펙터 쿼터니언, 유효한 회전을 나타내도록 반드시 정규화되어야 함 (ee_quat_base)
            - 로봇 베이스 프레임 기준 목표 위치 (target_pos_base)
        ]

    힌트: 쿼터니언 연산을 위해 제공된 함수 quat_mul, quat_conjugate, quat_normalize, rot_mat_to_quat를 사용할 수 있습니다.
    
    """
    raise NotImplementedError()
