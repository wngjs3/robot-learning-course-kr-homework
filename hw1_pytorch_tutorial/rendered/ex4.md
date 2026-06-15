# ex4 보기용 문서

> 이 파일은 GitHub에서 안정적으로 읽기 위한 Markdown 버전입니다. 직접 실행하려면 [`src/ex4.ipynb`](../src/ex4.ipynb) 노트북을 사용하세요.

# 실습 4: 이미지 분야에서의 트랜스포머 + GLU-MLP 소거 연구 (ViT × GLU 변형)

## 이번 실습에서는 두 가지 영향력 있는 아이디어를 결합합니다:

“An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale” (Dosovitskiy et al., 2020) https://arxiv.org/pdf/2010.11929 논문의 비전 트랜스포머 (Vision Transformers, ViT):
ViT는 이미지를 겹치지 않는 패치(예: 논문에서는 16×16)로 분할하고, 각 패치를 벡터로 임베딩한 후, 위치 정보를 추가하고, 분류를 위해 표준 트랜스포머 블록을 적용함으로써 이미지를 토큰 시퀀스처럼 처리할 수 있음을 보여줍니다.

“GLU Variants Improve Transformer” (Shazeer, 2020) https://arxiv.org/pdf/2002.05202 논문의 게이트형 MLP (Gated MLPs, GLU 변형):
Shazeer는 표준 트랜스포머 피드포워드 레이어(Feed-Forward Network, FFN / Multi-Layer Perceptron, MLP)를 GEGLU 및 SwiGLU와 같은 게이트형 선형 유닛(Gated Linear Unit, GLU) 변형으로 대체할 것을 제안합니다. 이는 종종 유사한 연산량/매개변수 예산 하에서 학습 동역학과 최종 성능을 향상시킵니다.

## 수행할 작업

MNIST를 위한 소형 ViT 스타일 분류기를 구현한 다음, 각 트랜스포머 블록 내부의 MLP를 대체하는 통제된 소거 연구(Ablation Study)를 실행합니다:

기준 모델(Baseline) FFN (GELU):
Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)

GLU 계열 MLP (최소 두 개를 선택하고 이유를 제시할 것):

GEGLU, SwiGLU, 또는 기타 활성화 함수(Activation Function)

여러분의 목표는 이러한 GLU 변형들이 다음 사항들을 변화시키는지 평가하는 것입니다:

- 수렴 속도 (손실(Loss) 대 스텝(Step)),

- 최종 테스트 정확도(Accuracy),

- 및/또는 여러 번의 실행에 걸친 안정성.

## 구현할 핵심 ViT 개념

- MNIST 이미지를 트랜스포머 토큰으로 변환하기 위해 다음을 수행합니다:
  각 28×28 이미지를 겹치지 않는 P×P 패치로 패치화(Patchify)합니다.
  만약 P=4라면, 7×7 패치 그리드를 얻게 되며 → 이미지당 49개의 토큰이 생성됩니다.

- 선형 레이어로 패치를 임베딩합니다: 패치 벡터 → d_model.

- 모델이 각 패치의 출처를 알 수 있도록 위치 임베딩(Positional Embedding)을 추가합니다.

- n_layers 개의 트랜스포머 인코더(Transformer Encoder) 블록을 적용합니다.

- 토큰 특징을 풀링(Pool)(예: 평균 풀링(Mean Pooling))하고 10개의 클래스로 투영(Project)합니다.

## 구현할 핵심 GLU 개념

GLU 스타일 MLP는 표준 FFN을 게이팅 메커니즘(Gating Mechanism)으로 대체합니다:
두 개의 투영 a와 b를 계산하고, a에 비선형 활성화 함수를 적용한 후(변형에 따라 다름), 원소별 곱셈(Elementwise Multiplication)인 act(a) * b를 수행하고, 다시 d_model로 투영합니다.
공정한 비교를 위해 Shazeer 논문의 2/3 가중치 폭 규칙(Width Rule)을 사용하십시오.

제공되는 사항 vs 직접 구현할 사항

### 제공되는 사항:

- MNIST 로딩 + 데이터로더(Dataloader)

- 최소한의 학습 루프(Training Loop) 구조 (AdamW)

- CPU에서 실행 가능한 권장 소형 모델 구성

### 직접 구현할 사항:

- 패치 토큰화 (patchify)

- 패치 임베딩 + 위치 임베딩 전략

- nn.MultiheadAttention을 사용하는 Pre-LN 트랜스포머 인코더 블록

- 최소 두 개의 GLU MLP 변형 + 하나의 FFN 기준 모델

- 결론을 뒷받침하기에 충분한 메트릭 기록(Logging)

## 자기 점검

최소 3가지 변형(기준 모델 + GLU를 위해 선택한 활성화 함수들)을 실행하고 다음을 보고하십시오:

- 최종 및 최고 테스트 정확도

- 학습 가능한 매개변수(Trainable Parameter) 수

- 에포크(Epoch)에 따른 손실/정확도 그래프 또는 요약 텍스트

- 결과에 대한 짧은 고찰(Discussion)

```python
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

```python
def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """이미지를 패치 토큰으로 변환합니다."""
    # TODO: 토큰화 전략 구현
    raise NotImplementedError
```

```python
# TODO: ViT 논문에서와 같이 positional encoding을 추가하고 patch projection을 구현하세요.
class PatchEmbed(nn.Module):
    def __init__(self, patch_dim: int, d_model: int):
        super().__init__()
        # TODO: 구현하세요.
        raise NotImplementedError

    def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
        # TODO: 구현하세요.
        raise NotImplementedError


class PositionalEmbedding(nn.Module):
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        # TODO: 구현하세요.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 구현하세요.
        raise NotImplementedError
```

```python
# TODO: GLU 논문에서 서로 비교하고자 하는 변형(variants)들을 정의하고, 그 선택 이유를 설명하세요.
class FeedForward(nn.Module):
    """
    표준 Transformer FFN:
      x -> Linear(d_model->d_ff) -> GELU -> Dropout -> Linear(d_ff->d_model) -> Dropout
    
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # TODO: 구현하세요
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 구현하세요
        raise NotImplementedError


class GLUFeedForward(nn.Module):
    """GLU 계열 FFN"""
    def __init__(self, d_model: int, d_ff_gated: int, dropout: float, variant: str):
        super().__init__()
        # TODO: 구현하세요
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 구현하세요
        raise NotImplementedError
```

```python
class TransformerEncoderBlock(nn.Module):
    """
    Pre-LN 인코더 블록:
      x = x + Dropout(SelfAttn(LN(x)))
      x = x + Dropout(MLP(LN(x)))
    
    """
    def __init__(self, d_model: int, n_heads: int, mlp: nn.Module, dropout: float):
        super().__init__()
        # TODO: 구현하기. 어텐션에는 nn.MultiHeadAttention을 사용하세요.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 구현하기
        raise NotImplementedError
```

```python
class TinyViT(nn.Module):
    """
    MNIST용 Tiny ViT 스타일 분류기.
    - patchify -> patch embed -> pos embed -> blocks -> mean pool -> head
    
    """
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        mlp_kind: str,
    ):
        super().__init__()
        assert 28 % patch_size == 0
        grid = 28 // patch_size
        self.num_tokens = grid * grid
        self.patch_size = patch_size
        patch_dim = patch_size * patch_size

        # TODO: 패치를 임베딩하는 전략을 구현하세요

        # TODO: 실험에 적합한 mlp 버전을 선택하는 전략을 구현하세요

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp=..., # TODO: mlp를 인코더 블록에 전달하세요
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # TODO: 출력 클래스 수로 프로젝션하는 헤드를 추가하세요

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 구현하세요
        logits = None
        return logits
```

```python
@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    batch_size: int = 128
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.01
    device: str = "cpu"  # 사용 가능한 경우 "cuda"로 설정
```

```python
def train_one_run(
    mlp_kind: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
) -> dict:
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_losses: list[float] = []
    test_accs: list[float] = []

    for epoch in range(cfg.epochs):

        # 학습 루프
        model.train()
        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            logits = model(xb)
            loss = ... # TODO: 손실 함수(criterion) 정의

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        # 평가 루프 NOTE: 이 부분을 변경할 필요는 없습니다
        model.eval()
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                logits = model(xb)
                correct += (logits.argmax(dim=-1) == yb).float().sum().item()
                total += yb.numel()

        test_accs.append(correct / total)
        print(f"[{mlp_kind}] epoch {epoch+1}/{cfg.epochs} | test acc: {test_accs[-1]:.4f}")

    return {
        # TODO: 이번 실험의 주장을 뒷받침할 수 있다고 생각하는 평가지표(metrics)를 반환하세요
    }
```

```python
cfg = TrainConfig(seed=0, batch_size=128, epochs=5, lr=3e-4, weight_decay=0.01, device="cpu")

tfm = transforms.Compose([transforms.ToTensor()])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

# 소형 모델 예시. TODO: 이 매개변수들을 자유롭게 실험해 보세요.
patch_size = 4
d_model = 64
n_heads = 4
n_layers = 2
d_ff = 256
dropout = 0.1

runs = [] # TODO: 실행(runs)의 이름을 지정하세요
results = []

for kind in runs:
    model = TinyViT(
        patch_size=patch_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        mlp_kind=kind,
    )
    # TODO: 여기에 출력하고 싶은 내용을 자유롭게 출력하세요
    print(f"\nRun: {kind} | " )
    out = train_one_run(kind, model, train_loader, test_loader, cfg)
    results.append(out)
```

