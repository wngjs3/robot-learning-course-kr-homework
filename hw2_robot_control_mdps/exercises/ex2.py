import numpy as np


def generate_quintic_spline_waypoints(start, end, num_points):

    """
    TODO:

    단계:
    1. 0과 1 사이에 선형 간격으로 배치된 `num_points`개의 시간 단계 `s`를 생성합니다.
    2. 슬라이드에 나와 있는 5차 시간 스케일링 다항식 함수를 적용하여 `f_s`를 구합니다.
    3. `start + (end - start) * f_s`를 사용하여 `start`와 `end` 사이를 보간합니다.
    
    Args:
        start (np.ndarray): 시작 웨이포인트.
        end (np.ndarray): 종료 웨이포인트.
        num_points (int): 궤적의 포인트 수.
        
    Returns:
        np.ndarray: 생성된 웨이포인트.
    
    """
    raise NotImplementedError()


def pid_control(tracking_error_history, timestep, Kp=150.0, Ki=0.0, Kd=0.01):
    """
    TODO:
    추적 오차 이력을 기반으로 PID 제어 신호를 계산합니다.
    
    단계:
    1. 비례(P) 항은 가장 최근의 오차입니다.
    2. 적분(I) 항은 모든 과거 오차의 합에 시뮬레이션 timestep을 곱한 값입니다.
    3. 미분(D) 항은 오차의 변화율(마지막 두 오차의 차이를 timestep으로 나눈 값)입니다.
       이전에 기록된 오차가 하나만 있는 경우, D 항은 0이어야 합니다.
    4. 최종 제어 신호를 계산합니다: Kp * P + Ki * I + Kd * D.
    
    Args:
        tracking_error_history (np.ndarray): 추적 오차 이력.
        timestep (float): 시뮬레이션 timestep.
        Kp (float): 비례 이득.
        Ki (float): 적분 이득.
        Kd (float): 미분 이득.
        
    Returns:
        np.ndarray: 제어 신호.
    
    """
    raise NotImplementedError()
            