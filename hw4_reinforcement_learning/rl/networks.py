from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    activation=nn.ReLU,
    output_activation=nn.Identity,
) -> nn.Sequential:
    """
    간단한 MLP를 구축합니다.
    
    """
    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    layers.append(output_activation())

    return nn.Sequential(*layers)


class ValueNet(nn.Module):
    """
    PPO에서 사용하는 상태 가치 네트워크 V(s)입니다.
    
    """

    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class QNet(nn.Module):
    """
    SAC에서 사용하는 행동 가치 네트워크 Q(s, a)입니다.
    
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.net = build_mlp(obs_dim + act_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)


class DoubleQNet(nn.Module):
    """
    두 개의 QNet으로 구성된 SAC용 Double Q-network입니다.
    
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.q1 = QNet(obs_dim, act_dim, hidden_sizes)
        self.q2 = QNet(obs_dim, act_dim, hidden_sizes)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q1_value = self.q1(obs, act)
        q2_value = self.q2(obs, act)
        return q1_value, q2_value


class GaussianActor(nn.Module):
    """
    PPO용 가우시안 정책입니다.

    출력은 다음과 같은 정규분포(Normal distribution)입니다:
        mean = network(obs)
        std  = exp(log_std), 여기서 log_std는 학습 가능한 파라미터 벡터입니다.
    
    """

    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.mu_net = build_mlp(obs_dim, hidden_sizes, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0, dtype=torch.float32))
        self.distribution = None

    def forward(self):
        # GaussianActor는 forward 패스를 지원하지 않습니다
        pass
        
    def update_distribution(self, obs: torch.Tensor):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std).expand_as(mu)
        self.distribution = Normal(mu, std)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
    SAC용 Squashed 가우시안 정책입니다.

    샘플링 파이프라인:
        u ~ Normal(mu, std)
        a = tanh(u)

    반환되는 action은 [-1, 1] 범위에 있습니다.
    
        """
        self.update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(action).sum(dim=-1)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mu_net(obs)


class SquashedGaussianActor(nn.Module):
    """
        반환값:
            action: 샘플링되었거나 결정론적인(deterministic) action
            log_prob: log pi(a|s)
        
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes[:-1], hidden_sizes[-1])
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.distribution = None

    def update_distribution(self, obs: torch.Tensor):
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        self.distribution = Normal(mu, std)

    def act(self, obs: torch.Tensor):
        """
        반환값:
            action: [-1, 1] 범위로 tanh-squashed된 action
            log_prob: 보정된 로그 확률(log-probability)
        
        """
        self.update_distribution(obs)
        raw_action = self.distribution.rsample()
        action = torch.tanh(raw_action)

        # tanh 보정을 적용하여 log_prob 계산
        log_prob = self.distribution.log_prob(raw_action)
        # log|d(action)/d(raw_action)| = log(1 - tanh(raw_action)^2)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)

        return action, log_prob.sum(dim=-1)
    
    def act_inference(self, obs: torch.Tensor):
        self.update_distribution(obs)
        action = self.distribution.mean
        action = torch.tanh(action)
        return action

    def forward(self):
        # SquashedGaussianActor는 forward 패스를 지원하지 않습니다
        pass