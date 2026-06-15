import gymnasium as gym
import numpy as np


class CartPoleWrapper:
    """
    Gymnasium의 CartPole 환경을 감싸는 경량 래퍼(wrapper)입니다.

    이 래퍼는 이번 과제를 위해 Gymnasium API를 다음과 같이 단순화합니다:
    - reset() -> state
    - step(action) -> next_state, reward, done, info

    또한 다음 작업을 수행합니다:
    - 관측값을 np.float32로 변환
    - `terminated`와 `truncated`를 하나의 `done` 플래그로 병합
    - `state_dim` 및 `action_dim`을 노출
    
    """

    def __init__(self, env_name="CartPole-v1", seed=0, render_mode=None):
        """
        환경을 생성합니다.

        Args:
            env_name (str): Gymnasium 환경의 이름
            seed (int): 재현성에 사용되는 무작위 시드
            render_mode (str 또는 None): 렌더링 모드 (예: None 또는 "rgb_array")
        
        """
        self.env_name = env_name
        self.seed = seed
        self.render_mode = render_mode

        # 기반이 되는 Gymnasium 환경을 생성합니다.
        self.env = gym.make(env_name, render_mode=render_mode)

        # 추후 사용을 위해 관측 공간(observation space)과 행동 공간(action space)을 저장합니다.
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # 재현성을 위해 무작위 시드(seed)를 설정합니다.
        self.env.reset(seed=seed)
        self.action_space.seed(seed)

        # CartPole의 경우:
        # - 관측(observation)은 4차원 벡터입니다.
        # - 행동 공간은 2개의 이산적(discrete) 행동을 가집니다.
        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.n

    def reset(self):
        """
        환경을 초기화하고 초기 상태를 반환합니다.

        Returns:
            np.ndarray: 초기 상태, 형태 (state_dim,), 데이터 타입 np.float32
        
        """
        state, _ = self.env.reset()
        return np.asarray(state, dtype=np.float32)

    def step(self, action):
        """
        환경에서 한 단계를 진행합니다.

        Args:
            action (int): 에이전트가 선택한 이산적 행동

        Returns:
            next_state (np.ndarray): 다음 관측값, 형태 (state_dim,)
            reward (float): 행동을 취한 후 받은 보상
            done (bool): 에피소드 종료 여부
            info (dict): Gymnasium에서 제공하는 추가 진단 정보
        
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)

        # 두 개의 Gymnasium 종료 시그널을 하나의 `done` 플래그로 병합합니다.
        done = terminated or truncated

        return (
            np.asarray(next_state, dtype=np.float32),
            float(reward),
            done,
            info,
        )

    def sample_action(self):
        """
        행동 공간에서 무작위 행동을 샘플링합니다.

        Returns:
            int: 무작위 이산적 행동
        
        """
        return int(self.action_space.sample())

    def close(self):
        """
        환경을 닫고 리소스를 해제합니다.
        
        """
        self.env.close()