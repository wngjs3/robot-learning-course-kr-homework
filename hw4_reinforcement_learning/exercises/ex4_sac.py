from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl.networks import DoubleQNet, SquashedGaussianActor
from rl.buffers import ReplayBatch


@dataclass
class SACUpdateStats:
    actor_loss: float | list[float]
    critic_loss: float | list[float]
    alpha_loss: float | list[float]
    alpha: float | list[float]

    @staticmethod
    def init_lists():
        return SACUpdateStats(
            actor_loss=[],
            critic_loss=[],
            alpha_loss=[],
            alpha=[],
        )

    def append(self, other) -> None:
        self.actor_loss.append(other.actor_loss)
        self.critic_loss.append(other.critic_loss)
        self.alpha_loss.append(other.alpha_loss)
        self.alpha.append(other.alpha)

    def mean(self):
        return SACUpdateStats(
            actor_loss=float(np.mean(self.actor_loss)),
            critic_loss=float(np.mean(self.critic_loss)),
            alpha_loss=float(np.mean(self.alpha_loss)),
            alpha=float(np.mean(self.alpha)),
        )


class SACAgent:
    """
    연속적 제어(continuous control)를 위한 Soft Actor-Critic (SAC) 에이전트.

    주요 구성 요소:
      - squashed Gaussian 정책 actor
      - 두 개의 Q-네트워크 (double Q-learning)
      - 두 개의 타겟 Q-네트워크
      - 자동 온도 조절을 포함한 엔트로피 정규화
    
    SAC 최적화 식:
      
    Critic 타겟:
        y = r + gamma * (1 - done) *
            [ min(Q1_target(s', a'), Q2_target(s', a')) - alpha * log pi(a'|s') ]

    Actor 목적 함수:
        J_pi = E[ alpha * log pi(a|s) - min(Q1(s, a), Q2(s, a)) ]

    온도(Temperature) 목적 함수:
        J_alpha = E[ -log_alpha * (log pi(a|s) + target_entropy) ]
    
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        init_alpha: float,
        target_entropy,
        device: torch.device,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.act_dim = act_dim

        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden_sizes).to(
            self.device
        )
        self.critic = DoubleQNet(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.critic_target = DoubleQNet(obs_dim, act_dim, hidden_sizes).to(
            self.device
        )

        # 초기에 타겟 critic으로 가중치 복사
        self.critic_target.load_state_dict(self.critic.state_dict())

        for p in self.critic_target.q1.parameters():
            p.requires_grad = False
        for p in self.critic_target.q2.parameters():
            p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        if target_entropy is None:
            target_entropy = -float(act_dim)
        self.target_entropy = target_entropy

        self.log_alpha = torch.tensor(
            np.log(init_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        """
        온도 파라미터 alpha = exp(log_alpha).
        
        """
        return self.log_alpha.exp()

    def sample_action(self, obs: torch.Tensor):
        """
        환경과의 상호작용을 위해 행동을 샘플링합니다.

        반환값:
            action (torch.Tensor): tanh squashing으로 인해 [-1, 1] 범위 내에 있는 행동
        
        """
        with torch.no_grad():
            # TODO: 환경과의 상호작용을 위해 actor로부터 행동을 샘플링하세요.
            # 
            # 힌트:
            # - self.actor.act(obs)는 (action, log_prob)을 반환합니다
            # - 여기서는 샘플링된 action만 필요합니다
            action = ...

        return action

    def predict_action(self, obs: torch.Tensor):
        """
        평가를 위한 결정론적(deterministic) 행동.
        
        """
        return self.actor.act_inference(obs)

    def compute_critic_loss(self, obs, act, rew, next_obs, done) -> torch.Tensor:
        """
        SAC critic 손실을 계산합니다.

        Critic 타겟:
            y = r + gamma * (1 - done) *
                [ min(Q1_target(s', a'), Q2_target(s', a')) - alpha * log pi(a'|s') ]

        인자:
            obs (torch.Tensor): 현재 관측값
            act (torch.Tensor): 취해진 행동
            rew (torch.Tensor): 보상
            next_obs (torch.Tensor): 다음 관측값
            done (torch.Tensor): 종료 여부 플래그

        반환값:
            torch.Tensor: critic 손실
        
        """
        with torch.no_grad():
            # TODO: SAC를 위한 타겟 Q 값을 계산하세요.
            # 
            # 힌트:
            # 1. 현재 actor로부터 next_action과 next_logp를 샘플링합니다
            # 2. self.critic_target을 사용하여 타겟 Q 값들을 계산합니다
            # 3. 두 타겟 critic 중 최솟값을 취합니다
            # 4. 엔트로피 정규화를 고려하기 위해 alpha * next_logp를 뺍니다
            # 5. 벨만 타겟을 구축합니다:
            #       rew + gamma * (1 - done) * q_next
            # 6. 두 Q-네트워크에 대해 타겟 Q 값과의 MSE 손실을 계산하고, 이를 평균하여 critic_loss를 구합니다
            next_action, next_logp = ...
            q1_next, q2_next = ...
            q_next = ...
            target_q = ...

        q1_pred, q2_pred = self.critic(obs, act)
        critic_loss = ...

        return critic_loss

    def compute_actor_loss(self, obs, act_new, logp_new) -> torch.Tensor:
        """
        SAC actor 손실을 계산합니다.

        Actor 목적 함수:
            J_pi = E[ alpha * log pi(a|s) - min(Q1(s, a), Q2(s, a)) ]

        인자:
            obs (torch.Tensor): 현재 관측값
            act_new (torch.Tensor): 현재 정책으로부터 새로 샘플링된 행동
            logp_new (torch.Tensor): 샘플링된 행동의 로그 확률

        반환값:
            torch.Tensor: actor 손실
        
        """
        # TODO: SAC actor 손실을 계산하세요.
        # 
        # 힌트:
        # 1. (obs, act_new)에서 두 critic을 모두 평가합니다
        # 2. 최소 Q 값을 취합니다
        # 3. 다음 목적 함수를 사용합니다:
        #       mean(alpha * logp_new - q_new)
        q1_new, q2_new = ...
        q_new = ...
        actor_loss = ...

        return actor_loss

    def compute_alpha_loss(self, logp_new) -> torch.Tensor:
        """
        SAC 온도(temperature) 손실을 계산합니다.

        온도 목적 함수:
            J_alpha = E[ -log_alpha * (log pi(a|s) + target_entropy) ]

        인자:
            logp_new (torch.Tensor): 새로 샘플링된 행동의 로그 확률

        반환값:
            torch.Tensor: alpha 손실
        
        """
        # TODO: SAC 온도(temperature) 손실을 계산하세요.
        # 
        # 힌트:
        # - self.alpha가 아닌 self.log_alpha를 사용하세요
        # - alpha 업데이트가 actor를 통해 역전파되지 않도록 (logp_new + target_entropy)를 detach하세요
        #   through the actor
        # - 배치에 대해 평균을 취합니다
        alpha_loss = ...

        return alpha_loss

    def soft_update_targets(self) -> None:
        """
        타겟 네트워크를 위한 Polyak 평균화:
            target <- tau * online + (1 - tau) * target
        
        """
        # TODO: 타겟 critic 파라미터를 소프트 업데이트하세요.
        # 
        # 힌트:
        # 각 파라미터 쌍에 대해:
        #   target_param <- (1 - tau) * target_param + tau * param
        with torch.no_grad():
            for target_param, param in zip(
                self.critic_target.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_( ... )

    def update(self, batch: ReplayBatch) -> SACUpdateStats:
        """
        하나의 SAC 업데이트 단계:
          1. critic 업데이트
          2. actor 업데이트
          3. alpha 업데이트
          4. 타겟 critic 소프트 업데이트

        인자:
            batch (ReplayBatch): 리플레이 버퍼에서 샘플링된 미니 배치

        반환값:
            SACUpdateStats: 이번 업데이트 단계의 통계치
        
        """
        obs = batch.obs
        act = batch.act
        rew = batch.rew
        next_obs = batch.next_obs
        done = batch.done

        # TODO: 하나의 SAC 업데이트 단계를 완료하세요.
        # 
        # 힌트:
        # 1. critic_loss를 계산하고 critic 파라미터를 업데이트합니다
        # 2. 현재 obs에 대해 actor로부터 새로운 행동들을 샘플링합니다
        # 3. actor_loss를 계산하고 actor 파라미터를 업데이트합니다
        # 4. alpha_loss를 계산하고 log_alpha를 업데이트합니다
        # 5. 타겟 critic들을 소프트 업데이트합니다
        critic_loss = ...
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        act_new, logp_new = ...

        actor_loss = ...
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = ...
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # TODO: 타겟 critic들을 소프트 업데이트합니다
        ...

        return SACUpdateStats(
            actor_loss=actor_loss.item(),
            critic_loss=critic_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=self.alpha.item(),
        )

    def save(self, path) -> None:
        """
        모델 파라미터 및 옵티마이저 상태를 저장합니다.
        
        """
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path) -> None:
        """
        모델 파라미터 및 옵티마이저 상태를 불러옵니다.
        
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

    def train_mode(self) -> None:
        """
        모듈들을 학습(training) 모드로 설정합니다.
        
        """
        self.actor.train()
        self.critic.train()
        self.critic_target.train()
        self.log_alpha.requires_grad_(True)

    def eval_mode(self) -> None:
        """
        모듈들을 평가(evaluation) 모드로 설정합니다.
        
        """
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()
        self.log_alpha.requires_grad_(False)