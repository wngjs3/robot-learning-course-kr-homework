"""SO-100 모방 정책을 위한 모델 정의."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """액션 청킹(action chunking) 정책의 기본 클래스."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """MSE 손실을 사용하여 액션 청크를 예측합니다.

    상태 벡터를 평탄화된 액션 청크(chunk_size * action_dim)로 매핑하고
    이를 (B, chunk_size, action_dim) 크기로 재구성하는 단순 MLP입니다.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """multicube 씬을 위한 목표 조건부(goal-conditioned) 정책."""
        raise NotImplementedError


# TODO: 학생들은 여기에 ObstaclePolicy를 구현하십시오.
class ObstaclePolicy(BasePolicy):
    """배치에 대한 학습 손실(training loss)을 계산합니다."""

    def forward(self) -> torch.Tensor:
        """(batch, chunk_size, action_dim) 크기의 액션 청크를 생성합니다."""
        raise NotImplementedError

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# TODO: 학생들은 여기에 MultiTaskPolicy를 구현하십시오.
class MultiTaskPolicy(BasePolicy):
    """(B, chunk_size, action_dim) 크기의 예측된 액션 청크를 반환합니다."""

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self) -> torch.Tensor:
        """(B, chunk_size, action_dim) 크기의 예측된 액션 청크를 반환합니다."""
        raise NotImplementedError


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    # TODO,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: 선택한 사양으로 빌드하십시오.
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: 선택한 사양으로 빌드하십시오.
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
