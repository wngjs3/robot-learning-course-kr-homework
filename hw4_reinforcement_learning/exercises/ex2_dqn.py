import random
import collections

import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    """
    하나의 은닉층(hidden layer)을 가진 Q-네트워크.

    입력:
        state

    출력:
        모든 이산 행동(discrete actions)에 대한 Q-값
    
    """

    def __init__(self, capacity):
        """
    이산 행동 공간(discrete action spaces)을 위한 심층 Q-네트워크(DQN).
    
        """
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        리플레이 버퍼를 초기화합니다.

        인자:
            capacity (int): 저장할 transition의 최대 개수
        
        """
        # TODO: 리플레이 버퍼에 transition을 추가하세요.
        raise NotImplementedError

    def sample(self, batch_size):
        """
        리플레이 버퍼에 하나의 transition을 저장합니다.

        인자:
            state (np.ndarray): 현재 상태
            action (int): 현재 상태에서 취한 행동
            reward (float): 행동을 취한 후 받은 보상
            next_state (np.ndarray): 다음 상태
            done (bool): 이 transition 이후 에피소드가 종료되는지 여부
        
        """
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        return (
            np.array(states, dtype=np.float32),
            actions,
            rewards,
            np.array(next_states, dtype=np.float32),
            dones,
        )

    def size(self):
        """
        무작위 미니배치 transition을 샘플링합니다.

        인자:
            batch_size (int): 샘플링할 transition의 개수

        반환값:
            states (np.ndarray): 크기 (batch_size, state_dim)
            actions (tuple): 길이 batch_size
            rewards (tuple): 길이 batch_size
            next_states (np.ndarray): 크기 (batch_size, state_dim)
            dones (tuple): 길이 batch_size
        
        """
        return len(self.buffer)


class QNet(torch.nn.Module):
    """
        현재 저장된 transition의 개수를 반환합니다.
        
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        Q-네트워크를 초기화합니다.

        인자:
            state_dim (int): 상태 공간의 차원
            hidden_dim (int): 은닉 유닛의 개수
            action_dim (int): 이산 행동의 개수
        
        """
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Q-네트워크의 forward 패스.

        인자:
            x (torch.Tensor): 크기 (batch_size, state_dim)

        반환값:
            torch.Tensor: 모든 행동에 대한 Q-값, 크기 (batch_size, action_dim)
        
        """
        # TODO: 네트워크의 forward 패스를 구현하세요.
        # 첫 번째 선형 레이어(linear layer) 뒤에 ReLU를 사용하세요.
        raise NotImplementedError


class DQN:
    """
        DQN 에이전트를 초기화합니다.

        인자:
            state_dim (int): 상태 공간의 차원
            hidden_dim (int): Q-네트워크의 은닉 차원
            action_dim (int): 이산 행동의 개수
            learning_rate (float): Adam의 학습률
            gamma (float): 할인 요인(discount factor)
            epsilon (float): epsilon-greedy 정책의 탐색 확률
            target_update (int): 타겟 네트워크 업데이트 주기
            device (torch.device): cpu 또는 cuda
        
    """

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        """
        epsilon-greedy 정책을 사용하여 행동을 선택합니다.

        epsilon 확률로 무작위 행동을 선택합니다.
        그렇지 않으면 예측된 Q-값이 가장 높은 행동을 선택합니다.

        인자:
            state (np.ndarray): 현재 상태, 크기 (state_dim,)

        반환값:
            int: 선택된 행동
        
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device

        # 온라인 Q-네트워크
        self.q_net = QNet(state_dim, hidden_dim, action_dim).to(device)

        # 타겟 Q-네트워크
        self.target_q_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # 옵티마이저
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        # 주기적인 타겟 네트워크 업데이트에 사용되는 카운터
        self.count = 0

    def take_action(self, state):
        """
        그리디(greedy) 행동을 선택합니다.

        이 함수는 평가(evaluation) 중에 유용합니다.

        인자:
            state (np.ndarray): 현재 상태

        반환값:
            int: 그리디 행동
        
        """
        # TODO: epsilon-greedy 행동 선택을 구현하세요.
        # 힌트:
        # - np.random.random()을 사용하여 탐색(explore) 여부를 결정하세요.
        # - 활용(exploitation)의 경우, 상태를 (1, state_dim) 크기의 torch 텐서로 변환하고,
        #   이를 `self.device`로 이동시킨 후 가장 큰 Q-값을 가진 행동을 선택하세요.
        # 현재 Q 값 계산
        raise NotImplementedError

    def predict_action(self, state):
        """
        한 미니배치의 transition을 사용하여 온라인 Q-네트워크를 업데이트합니다.

        인자:
            transition_dict (dict): 다음 항목을 포함
                - 'states'
                - 'actions'
                - 'rewards'
                - 'next_states'
                - 'dones'
        
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.q_net(state).argmax().item()
        return int(action)

    def update(self, transition_dict):
        """
        모델 파라미터를 저장합니다.
        
        """
        states = torch.tensor(
            transition_dict["states"], dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor(
            transition_dict["actions"], dtype=torch.long
        ).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            transition_dict["rewards"], dtype=torch.float32
        ).view(-1, 1).to(self.device)
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float32
        ).to(self.device)
        dones = torch.tensor(
            transition_dict["dones"], dtype=torch.float32
        ).view(-1, 1).to(self.device)

        # TD 타겟 계산
        q_values = self.q_net(states).gather(1, actions)

        # TODO: TD 타겟 `q_targets`를 계산하세요.
        with torch.no_grad():
            # 힌트:
            # - 다음 상태(next-state) 값에는 타겟 네트워크를 사용하세요.
            # - DQN 타겟: r + gamma * max_a' Q_target(s', a') * (1 - done)
            # DQN 손실(loss) 계산
            raise NotImplementedError

        # Q-네트워크 최적화
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        # 주기적으로 타겟 네트워크 업데이트
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        #      오프폴리시(off-policy) 강화학습을 위한 경험 리플레이 버퍼(experience replay buffer).      버퍼는 다음과 같은 형태의 transition을 저장합니다:         (state, action, reward, next_state, done)      학습 중에는 연속된 transition 간의 시간적 상관관계를 깨기 위해     버퍼에서 무작위로 미니배치를 샘플링합니다.     
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1

    def save(self, path):
        """
        모델 파라미터를 불러옵니다.
        
        """
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_q_net": self.target_q_net.state_dict(),
            },
            path,
        )

    def load(self, path):
        """
        Load model parameters.
        
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_q_net.load_state_dict(checkpoint["target_q_net"])