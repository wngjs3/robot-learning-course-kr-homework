import copy
import numpy as np


class PolicyIteration:
    """
    유한 테이블 형식 MDP에 대한 정책 반복(Policy iteration).

    Attributes:
        env:
            환경 객체. 다음을 가정함:
                - env.n_states: int
                - env.n_actions: int
                - env.P[s][a] = [(prob, next_state, reward, done)]
        theta: float
            정책 평가의 수렴 임계값.
        gamma: float
            할인율(Discount factor).
        v: np.ndarray, 형상 (n_states,)
            상태 가치 함수.
        pi: np.ndarray, 형상 (n_states, n_actions)
            확률적 정책. 각 행 pi[s]는 상태 s에서 행동들에 대한 확률 분포임.
    
    """

    def __init__(self, env, theta=1e-3, gamma=0.9):
        """
    유한 테이블 형식 MDP에 대한 가치 반복(Value iteration).

    Attributes:
        env:
            테이블 형식 전이 모델 env.P를 가진 환경 객체.
        theta: float
            수렴 임계값.
        gamma: float
            할인율(Discount factor).
        v: np.ndarray, 형상 (n_states,)
            상태 가치 함수.
        pi: np.ndarray, 형상 (n_states, n_actions)
            수렴된 가치 함수로부터 추출된 탐욕적(Greedy) 정책.
    
        """
        self.env = env
        self.theta = theta
        self.gamma = gamma

        # 가치 함수를 0으로 초기화합니다.
        self.v = np.zeros(self.env.n_states, dtype=float)

        # 정책을 균등한 무작위 분포로 초기화합니다.
        self.pi = np.ones((self.env.n_states, self.env.n_actions), dtype=float)
        self.pi /= self.env.n_actions

    def policy_evaluation(self):
        """정책 반복을 초기화합니다."""
        while True:
            max_diff = 0.0
            new_v = np.zeros_like(self.v)

            for s in range(self.env.n_states):
                qsa_list = []
                for a in range(self.env.n_actions):
                    qsa = 0.0
                    
                    # TODO: 현재 정책 하에서 상태 s의 업데이트된 가치를 계산하세요.
                    # 
                    # 권장 단계:
                    # 1. 각 행동 a에 대해, self.v 하에서의 행동 가치를 계산합니다.
                    # 2. q_pi(s, a)에 pi[s][a]로 가중치를 부여합니다.
                    # 3. 모든 행동에 대해 합산하여 new_v[s]를 얻습니다.
                    raise NotImplementedError("TODO: implement policy evaluation update")
                
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))

            self.v = new_v

            # TODO: 가치 함수가 수렴하면 중단하세요.
            raise NotImplementedError("TODO: add convergence check")

    def policy_improvement(self):
        """
        수렴할 때까지 현재 정책을 평가합니다.

        Input:
            사용 항목:
                - self.pi: np.ndarray, 형상 (n_states, n_actions)
                - self.v:  np.ndarray, 형상 (n_states,)
                - self.env.P

        Output:
            업데이트 항목:
                - self.v: np.ndarray, 형상 (n_states,)
        
        """
        for s in range(self.env.n_states):
            qsa_list = []
            for a in range(self.env.n_actions):
                qsa = 0.0
                
                # TODO: 상태 s에서 모든 행동에 대한 qsa_list를 계산하세요.
                raise NotImplementedError("TODO: compute q-values for policy improvement")

            max_q = max(qsa_list)
            num_best_actions = sum(np.isclose(qsa_list, max_q))
            self.pi[s] = [
                1.0 / num_best_actions if np.isclose(q, max_q) else 0.0
                for q in qsa_list
            ]

        return self.pi

    def policy_iteration(self):
        """
        현재 가치 함수에 대해 탐욕적(greedy)으로 현재 정책을 개선합니다.

        Input:
            사용 항목:
                - self.v: np.ndarray, 형상 (n_states,)
                - self.env.P

        Output:
            업데이트 및 반환 항목:
                - self.pi: np.ndarray, 형상 (n_states, n_actions)

        모든 탐욕적 행동에 동일한 확률을 할당합니다 (동률 허용).
        
        """
        while True:
            old_pi = copy.deepcopy(self.pi)

            # TODO: 정책 반복(policy iteration)의 메인 루프를 구현하세요.
            raise NotImplementedError("TODO: implement policy iteration main loop")

            if np.allclose(old_pi, new_pi):
                break

        return self.v, self.pi


class ValueIteration:
    """
        정책이 더 이상 변경되지 않을 때까지 정책 반복을 실행합니다.

        Input:
            사용 항목:
                - env.P
                - self.theta
                - self.gamma

        Output:
            v:  np.ndarray, 형상 (n_states,)
                최종 수렴된 가치 함수.
            pi: np.ndarray, 형상 (n_states, n_actions)
                최종 개선된 정책.

        메인 루프:
            1. 정책 평가 (Policy evaluation)
            2. 정책 개선 (Policy improvement)
            3. 정책이 변경되지 않으면 중단
        
    """

    def __init__(self, env, theta=1e-3, gamma=0.9):
        """가치 반복을 초기화합니다."""
        self.env = env
        self.theta = theta
        self.gamma = gamma

        self.v = np.zeros(self.env.n_states, dtype=float)
        self.pi = np.zeros((self.env.n_states, self.env.n_actions), dtype=float)

    def value_iteration(self):
        """
        수렴할 때까지 가치 반복을 실행합니다.

        Input:
            사용 항목:
                - self.v: np.ndarray, 형상 (n_states,)
                - self.env.P

        Output:
            반환 항목:
                - v:  np.ndarray, 형상 (n_states,)
                - pi: np.ndarray, 형상 (n_states, n_actions)
        
        """
        while True:
            max_diff = 0.0
            new_v = np.zeros_like(self.v)

            for s in range(self.env.n_states):
                qsa_list = []
                for a in range(self.env.n_actions):
                    qsa = 0.0
                    
                    # TODO: 모든 행동 가치 Q(s, a)를 계산하세요.
                    raise NotImplementedError("TODO: implement value iteration update")

                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))

            self.v = new_v
            
            # TODO: 가치 함수가 수렴하면 중단하세요.
            raise NotImplementedError("TODO: add convergence check")

        self.get_policy()
        return self.v, self.pi

    def get_policy(self):
        """
        수렴된 가치 함수로부터 탐욕적 정책을 추출합니다.

        Input:
            사용 항목:
                - self.v: np.ndarray, 형상 (n_states,)

        Output:
            업데이트 항목:
                - self.pi: np.ndarray, 형상 (n_states, n_actions)
        
        """
        for s in range(self.env.n_states):
            qsa_list = []
            for a in range(self.env.n_actions):
                qsa = 0.0
                # TODO: 모든 행동에 대한 qsa_list를 계산하세요.
                raise NotImplementedError("TODO: compute q-values for greedy policy extraction")

            max_q = max(qsa_list)
            num_best_actions = sum(np.isclose(qsa_list, max_q))
            self.pi[s] = [
                1.0 / num_best_actions if np.isclose(q, max_q) else 0.0
                for q in qsa_list
            ]