import numpy as np
import mujoco


def get_lemniscate_keypoint(t, a=0.2):
    """
    TODO:
    Y-Z 평면에서 베르누이 레미니스케이트(무한대 기호)를 사용하여 키포인트 세트를 생성합니다.
        공식: y = a * cos(t) / (1 + sin(t)^2)
              z = a * cos(t) * sin(t) / (1 + sin(t)^2)
    참고로, 베르누이 레미니스케이트에 대한 자세한 내용은 위키피디아에서 확인할 수 있습니다: https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli
    
    Args:
        t (float 또는 np.ndarray): 키포인트를 생성하기 위한 0에서 2π 사이의 시간 스케일.
        a (float): 레미니스케이트 크기의 스케일링 인자.
        
    Returns:
        y (float 또는 np.ndarray): 레미니스케이트 위 키포인트의 y 좌표.
        z (float 또는 np.ndarray): 레미니스케이트 위 키포인트의 z 좌표.
    
    """
    raise NotImplementedError()

def build_keypoints(count=16, width=0.25, x_offset=0.3, z_offset=0.25):
    """TODO:
    레미니스케이트 궤적을 따라 키포인트 세트(x, y, z)를 구축합니다.
    단계:
    1. 0과 2π(제외) 사이에 균일한 간격의 시간 값 `t`를 `count` 개수만큼 생성합니다.
    2. 각 시간 값 `t`에 대해 `get_lemniscate_keypoint(t, a=width)`를 사용하여 해당하는 (y, z) 좌표를 계산합니다.
    3. (y, z) 좌표를 고정된 x 좌표(x_offset) 및 가산되는 z_offset과 결합하여 [x_offset, y, z + z_offset] 형식의 3D 키포인트를 생성합니다.
    4. 키포인트를 (count, 3) 형상의 NumPy 배열로 반환합니다.

    Args:
        count (int): 궤적을 따라 생성할 키포인트 수.
        width (float): 레미니스케이트 크기의 스케일링 인자.
        x_offset (float): 모든 키포인트의 고정된 x 좌표.
        z_offset (float): 모든 키포인트의 z 좌표에 더해질 오프셋.

    Returns:
        np.ndarray: 생성된 키포인트를 포함하는 (count, 3) 형상의 배열.
    
    """
    raise NotImplementedError()

def ik_track(model, data, site_name, target_pos,
             damping=1e-3, pos_gain=2.0, dt=0.1, max_iters=2000):
    """TODO:
    목표 말단 장치 위치에 도달하기 위한 관절 설정을 계산하는 IK 추적 함수를 구현합니다. 단순화를 위해 방향 추적은 무시합니다.
    이 함수는 지정된 허용 오차 내에서 목표에 도달하거나 최대 반복 횟수를 초과할 때까지 말단 장치의 자코비안을 사용하여 관절 설정을 반복적으로 업데이트해야 합니다.
    자코비안의 특이점(singularity)을 처리하기 위해 감쇠 최소제곱법을 사용합니다. 참고로, 감쇠 최소제곱법에 대한 자세한 내용은 위키피디아에서 확인할 수 있습니다: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

    단계:
    1. 나중에 복원할 원래 관절 설정(qpos)을 저장합니다.
    2. 최대 반복 횟수 동안 반복:
        a. 순운동학(mj_kinematics)을 사용하여 현재 말단 장치 위치와 방향을 계산합니다.
        b. 위치 오차(target_pos - current_pos)를 계산합니다.
        c. 위치 오차가 특정 임계값(예: 1e-3) 미만이면 목표에 도달했으므로 루프를 탈출합니다.
        d. mj_jacSite를 사용하여 말단 장치의 자코비안을 계산합니다.
        e. 감쇠 최소제곱법을 사용하여 오차를 줄일 수 있는 관절 설정의 변화량(qdot)을 계산합니다.
        f. 감쇠 최소제곱법의 출력을 사용하여 관절 설정(qpos)을 업데이트합니다.
    3. 원래 관절 설정을 복원하고 계산된 목표 관절 설정을 반환합니다.

    Args:
        model: MuJoCo 모델 객체.
        data: MuJoCo 데이터 객체.
        site_name: 추적할 말단 장치 사이트의 이름.
        target_pos: 말단 장치의 목표 위치 (3D 벡터).
        damping: 특이점 처리를 위한 감쇠 최소제곱법의 감쇠 인자.
        pos_gain: 제어 신호에서 위치 오차에 대한 이득(gain) 인자.
        dt: 관절 설정 업데이트를 위한 시간 단계(time step).
        max_iters: 목표 도달을 시도할 최대 반복 횟수.

    Returns:
        np.ndarray: 원하는 말단 장치 위치를 달성하는 목표 관절 설정(qpos).
    
    """
    num_joints = model.nv
    # 나중에 복원할 원래 관절 설정(joint configuration)을 저장
    original_qpos = data.qpos.copy()

    for i in range(max_iters):
        # 순운동학(forward kinematics)을 사용하여 현재 말단 장치(end-effector) 위치를 업데이트: data.site(site_name).xpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        # TODO: 말단 장치 위치 오차 계산
        err_pos = ...

        # TODO: 위치 오차의 2-norm이 작은 임계값(1e-3) 이내인지 확인하고, 맞다면 루프 탈출
        ...
        
        # mj_jacSite를 사용하여 말단 장치의 자코비안(Jacobian) 획득
        jacp = np.zeros((3, num_joints)) # 위치 자코비안
        jacr = np.zeros((3, num_joints)) # 방향 자코비안
        mujoco.mj_jacSite(model, data, jacp, jacr, model.site(site_name).id)
        J = np.vstack([jacp, jacr])  # 형상 (6, nv)

        # TODO: 위치 오차를 줄이기 위해 감쇠 최소제곱법(Damped Least Squares method)을 사용하여 관절 설정의 변화량(qdot) 계산
        # 감쇠 최소제곱법: qdot = J^T @ (J @ J^T + damping * I)^-1 @ weighted_err
        # 힌트: damping * I는 대각선에 damping이 있는 6x6 행렬이며, weighted error는 다음과 같은 형태의 6D 벡터(위치 3개, 회전 3개)입니다.
        # [pos_gain * err_pos, rot_gain * err_rot]. 방향 추적은 무시하므로 weighted error의 회전 성분은 0으로 설정할 수 있습니다.
        # 행렬 역행렬을 직접 계산하는 대신(수치적으로 불안정할 수 있음), np.linalg.solve를 사용하여
        # 선형 시스템 (J @ J^T + damping * I) x = weighted_err의 x를 풀고, qdot = J^T @ x를 계산해야 합니다. 이는 역행렬을 계산하는 것보다 더 안정적이고 효율적입니다.
        qdot = ...

        # 오버슈트를 방지하기 위한 선택적 클램핑(clamp)
        qdot = np.clip(qdot, -2.0, 2.0)

        # 감쇠 최소제곱법의 출력을 사용하여 관절 설정(qpos) 업데이트
        data.qvel[:] = 0.0
        data.qpos[:] += qdot * dt

    # 목표에 도달하지 못하고 루프를 종료하는 경우 경고 메시지 출력
    if i >= max_iters - 1 and np.linalg.norm(err_pos) >= 5e-3:
        print("Warning: IK did not converge within the iteration limit.")
        print(f"Final position error: {np.linalg.norm(err_pos):.4f}")

    # 원래 관절 설정을 복원하고 목표 관절 설정을 반환
    target_qpos = data.qpos.copy()
    data.qpos[:] = original_qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_forward(model, data)
    return target_qpos
