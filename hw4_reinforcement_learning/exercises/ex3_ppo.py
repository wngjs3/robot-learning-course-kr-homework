from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from collections.abc import Generator

from rl.networks import GaussianActor, ValueNet
from rl.buffers import RolloutBatch


@dataclass
class PPOUpdateStats:
    mean_kl: float
    mean_surrogate_loss: float
    mean_value_loss: float
    mean_entropy: float


class PPOAgent:
    """
    연속적인 행동 공간을 위한 Proximal Policy Optimization (PPO) 에이전트.
    
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes,
        n_steps: int = 2048,
        mini_batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        surrogate_loss_coeff: float = 1.0,
        value_loss_coeff: float = 1.0,
        entropy_coeff: float = 0.0,
        clip_ratio: float = 0.2,
        learning_rate: float = 1.0e-3,
        target_kl: float = 0.05,
        max_grad_norm: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.n_steps = n_steps
        self.mini_batch_size = mini_batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.surrogate_loss_coeff = surrogate_loss_coeff
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_ratio = clip_ratio
        self.learning_rate = learning_rate
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.actor = GaussianActor(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.critic = ValueNet(obs_dim, hidden_sizes).to(self.device)

        # actor 및 critic을 위한 통합 옵티마이저
        self.optimizer = optim.Adam(
            chain(self.actor.parameters(), self.critic.parameters()),
            lr=self.learning_rate,
        )

    def select_action(self, obs: torch.Tensor):
        """
        현재 정책으로부터 행동을 샘플링합니다.

        Args:
            obs (torch.Tensor): 관측 텐서

        Returns:
            action (torch.Tensor): 샘플링된 행동
            action_clipped (torch.Tensor): [-1, 1] 범위로 클리핑된 행동
            value (float): critic 예측값 V(s)
            action_log_prob (float): 로그 확률 log pi(a|s)
            action_mu (torch.Tensor): 가우시안 정책의 평균
            action_std (torch.Tensor): 가우시안 정책의 표준편차
        
        """
        with torch.inference_mode():
            # TODO: actor로부터 행동을 샘플링하고 이에 대응하는
            # 출력을 계산하십시오.
            # 
            # 수행해야 할 작업:
            # 1. self.actor.act(obs)로 행동 샘플링
            # 2. 행동을 [-1, 1] 범위로 클리핑
            # 3. 행동의 로그 확률 계산
            # 4. 현재 정책의 평균(mean)과 표준편차(std) 읽기
            # 5. critic으로부터 상태 가치 계산
            action = ...
            action_clipped = ...
            action_log_prob = ...
            action_mu = ...
            action_std = ...
            value = ...

        return action, action_clipped, value, action_log_prob, action_mu, action_std

    def predict_action(self, obs: torch.Tensor):
        """
        평가를 위한 결정론적(deterministic) 행동.
        
        """
        action = self.actor.act_inference(obs)
        return torch.clamp(action, -1.0, 1.0)

    def compute_kl_mean(self, old_mu_batch, old_std_batch, mu_batch, std_batch):
        """
        두 가우시안 행동 분포 간의 평균 KL divergence를 계산합니다.

        Args:
            old_mu_batch (torch.Tensor): 이전 정책의 평균
            old_std_batch (torch.Tensor): 이전 정책의 표준편차
            mu_batch (torch.Tensor): 새 정책의 평균
            std_batch (torch.Tensor): 새 정책의 표준편차

        Returns:
            torch.Tensor: 스칼라 평균 KL divergence
        
        """
        # TODO: 두 가우시안 행동 분포 간의 KL divergence를 구현하십시오.
        # 
        # 힌트:
        # 각 행동 차원별:
        #   KL = log(std / old_std)
        #        + (old_std^2 + (old_mu - mu)^2) / (2 * std^2)
        #        - 0.5
        # 
        # 그 다음:
        # - 행동 차원에 대해 합산(sum)
        # - 미니배치에 대해 평균(average)
        kl_per_dim = ...
        kl_per_sample = ...
    
        return kl_per_sample.mean()
        

    def adjust_learning_rate(self, kl, current_lr, min_lr=1e-5, max_lr=1e-3):
        """
        KL divergence에 따라 학습률을 조정합니다.
        
        """
        new_lr = current_lr
        if kl > self.target_kl * 2.0:
            new_lr = max(current_lr / 1.5, min_lr)
        elif kl < self.target_kl / 1.5 and kl > 0:
            new_lr = min(current_lr * 1.5, max_lr)
        return new_lr

    def compute_surrogate_loss(self, logp_batch, old_logp_batch, adv_batch):
        """
        PPO clipped surrogate loss를 계산합니다.

        Args:
            logp_batch (torch.Tensor): 새 로그 확률
            old_logp_batch (torch.Tensor): 이전 로그 확률
            adv_batch (torch.Tensor): 어드밴티지(advantage) 추정치

        Returns:
            torch.Tensor: 스케일링된 surrogate loss
        
        """
        # TODO: PPO clipped surrogate objective를 구현하십시오.
        # 
        # 힌트:
        # 1. ratio = exp(new_logp - old_logp)
        # 2. clipped_ratio = clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        # 3. objective = min(ratio * adv, clipped_ratio * adv)
        # 4. PPO는 손실(loss)을 최소화하므로, 음의 평균 objective를 사용하십시오.
        ratio = ...
        clipped_ratio = ...
        surrogate_loss = ...
        
        return self.surrogate_loss_coeff * surrogate_loss

    def compute_value_loss(self, val_batch, old_val_batch, ret_batch):
        """
        클리핑이 적용된 value loss를 계산합니다.
        
        """
        # TODO: 클리핑이 적용된 PPO value loss를 구현하십시오.
        # 
        # 힌트:
        # 1. 클리핑되지 않은 value loss 계산: (val - ret)^2
        # 2. 가치 예측 클리핑:
        #    old_val + clamp(val - old_val, -clip_ratio, clip_ratio)
        # 3. 클리핑된 손실 계산
        # 4. 클리핑된 손실과 클리핑되지 않은 손실 중 최댓값 선택
        # 5. 평균을 구하고 value_loss_coeff로 스케일링
        value_loss_unclipped = ...
        value_clipped = ...
        value_loss_clipped = ...
        value_loss = ...
        
        return self.value_loss_coeff * value_loss

    def compute_entropy_loss(self, entropy_batch):
        """
        엔트로피 정규화 항을 계산합니다.
        
        """
        # TODO: PPO entropy loss를 구현하십시오.
        # 힌트: PPO는 엔트로피를 최대화합니다.
        return ...

    def mini_batch_generator(self, batch) -> Generator:
        """
        PPO 업데이트를 위한 데이터 미니배치를 생성합니다.
        
        """
        for _ in range(self.n_epochs):
            indices = torch.randperm(
                self.n_steps, requires_grad=False, device=self.device
            )
            for start in range(0, self.n_steps, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                yield RolloutBatch(
                    obs=batch.obs[batch_indices],
                    act=batch.act[batch_indices],
                    logp=batch.logp[batch_indices],
                    mu=batch.mu[batch_indices],
                    std=batch.std[batch_indices],
                    val=batch.val[batch_indices],
                    ret=batch.ret[batch_indices],
                    adv=batch.adv[batch_indices],
                )

    def update(self, rollout_batch) -> PPOUpdateStats:
        """
        전체 롤아웃 배치를 사용하여 PPO actor 및 critic을 업데이트합니다.

        Args:
            rollout_batch: 환경과의 상호작용을 통해 수집된 전체 배치

        Returns:
            PPOUpdateStats: 모든 미니배치 업데이트에 대해 평균을 낸 통계치
        
        """
        mean_kl = 0
        mean_surrogate_loss = 0
        mean_value_loss = 0
        mean_entropy = 0
        num_updates = 0

        for mini_batch in self.mini_batch_generator(rollout_batch):
            obs_batch = mini_batch.obs
            act_batch = mini_batch.act
            old_logp_batch = mini_batch.logp
            old_mu_batch = mini_batch.mu
            old_std_batch = mini_batch.std
            old_val_batch = mini_batch.val
            ret_batch = mini_batch.ret
            adv_batch = mini_batch.adv

            self.actor.update_distribution(obs_batch)
            logp_batch = self.actor.get_actions_log_prob(act_batch)
            mu_batch = self.actor.action_mean
            std_batch = self.actor.action_std
            val_batch = self.critic(obs_batch)
            entropy_batch = self.actor.entropy

            # TODO: 하나의 PPO 업데이트 단계를 완료하십시오.
            # 
            # 수행해야 할 작업:
            # 1. 이전 정책과 새 정책 간의 KL divergence 계산
            # 2. 학습률을 조정하고 optimizer.param_groups 업데이트
            # 3. surrogate loss 계산
            # 4. value loss 계산
            # 5. entropy loss 계산
            # 6. 이들을 합산하여 최종 손실 계산
            # 7. zero grad, backward, gradient clipping, optimizer step 수행
            kl = ...
            self.learning_rate = ...
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate
            surrogate_loss = ...
            value_loss = ...
            entropy_loss = ...
            loss = ...

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()

            mean_kl += kl
            mean_surrogate_loss += surrogate_loss.item()
            mean_value_loss += value_loss.item()
            mean_entropy += entropy_batch.mean().item()
            num_updates += 1

        mean_kl /= num_updates
        mean_surrogate_loss /= num_updates
        mean_value_loss /= num_updates
        mean_entropy /= num_updates

        return PPOUpdateStats(
            mean_kl=mean_kl,
            mean_surrogate_loss=mean_surrogate_loss,
            mean_value_loss=mean_value_loss,
            mean_entropy=mean_entropy,
        )

    def save(self, path) -> None:
        """
        모델 파라미터와 옵티마이저 상태를 저장합니다.
        
        """
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path) -> None:
        """
        모델 파라미터와 옵티마이저 상태를 불러옵니다.
        
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def train_mode(self) -> None:
        """
        actor와 critic을 훈련(training) 모드로 설정합니다.
        
        """
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        """
        actor와 critic을 평가(evaluation) 모드로 설정합니다.
        
        """
        self.actor.eval()
        self.critic.eval()