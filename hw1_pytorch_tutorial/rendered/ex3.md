# ex3 보기용 문서

> 이 파일은 GitHub에서 안정적으로 읽기 위한 Markdown 버전입니다. 직접 실행하려면 [`src/ex3.ipynb`](../src/ex3.ipynb) 노트북을 사용하세요.

# 과제 3: PyTorch를 이용한 신경망 (Neural networks in PyTorch)

이 과제에서는 신경망(Neural Network)을 구성하는 작은 블록들을 처음부터 직접 구현하고, 이를 사용하여 간단한 분류기(Classifier)를 학습시킵니다.

다룰 내용은 다음과 같습니다:
- **기본 레이어(Basic layers)**: 선형(Linear), 임베딩(Embedding), 드롭아웃(Dropout)
- **정규화(Normalization)**: 레이어 정규화(LayerNorm) 및 RMS 정규화(RMSNorm)
- **MLP + 잔차 연결(Residual)**: 레이어들을 조합하여 더 깊은 네트워크 구성하기
- **분류(Classification)**: 학습 가능한 데이터셋 생성, 로짓(Logits)으로부터 교차 엔트로피(Cross-Entropy) 구현, 최소한의 학습 루프(Training loop) 작성

이전과 마찬가지로: 함수 이름이나 시그니처를 변경하지 않고 모든 `TODO`를 채우십시오.
간단한 정상성 검사(Sanity check)를 사용하고, 필요한 경우 PyTorch 레퍼런스 구현과 비교해 보세요.

```python
from __future__ import annotations

import torch
from torch import nn
```

## 기본 레이어 (Basic layers)

이 섹션에서는 어디서나 사용되는 몇 가지 핵심 레이어를 구현합니다:

### `Linear`
`nn.Linear` 규약을 따르는 완전 연결 레이어(Fully-connected layer)입니다:  
`y = x @ Wᵀ + b`

중요 세부 사항:
- 매개변수(Parameter)는 `nn.Parameter`로 등록되어야 합니다.
- 가중치(Weight)는 `nn.Linear`와 같이 (out_features, in_features) 형태로 저장합니다.
- 순방향 패스(Forward pass)는 선두의 배치 차원을 지원해야 합니다: `x`는 `(..., in_features)` 형태를 가질 수 있습니다.

### `Embedding`
임베딩 테이블은 정수 ID를 벡터에 매핑합니다:
- 입력: `(...,)` 형태의 토큰 ID `idx`
- 출력: `(..., embedding_dim)` 형태의 벡터

이것은 본질적으로 학습 가능한 룩업 테이블(Lookup table)입니다.

### `Dropout`
드롭아웃은 과적합(Overfitting)을 줄이기 위해 학습 중에 무작위로 활성화(Activation)를 0으로 만듭니다.
구현 세부 사항:
- `model.train()` 모드에서만 활성화됩니다.
- 학습 시: `p` 확률로 값을 누락(Drop)시키고, 유지된 값들을 `1/(1-p)`로 스케일링하여 기댓값이 동일하게 유지되도록 합니다.
- 평가(Eval) 시: 입력을 변경 없이 그대로 반환합니다.

## 지침
- 직접 구현하는 부분에는 PyTorch 레퍼런스 모듈을 사용하지 마십시오 (예: `Linear` 내부에서 `nn.Linear`를 호출하지 마십시오).
- 이전에 배운 표준 텐서 연산(matmul, sum, mean, rsqrt, 인덱싱 등)을 사용할 수 있습니다.
- 원하는 매개변수 초기화 방법을 사용하십시오. Xavier-uniform과 같은 방법을 권장합니다.

```python
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # TODO: 파라미터 초기화
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        return: (..., out_features)
        
        """
        # TODO: 구현
        raise NotImplementedError
```

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # TODO: 초기화
        raise NotImplementedError

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (...,) int64
        return: (..., embedding_dim)
        
        """
        # TODO: 구현 (가중치 인덱싱)
        raise NotImplementedError
```

```python
class Dropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        train 모드: p의 확률로 드롭하고 1/(1-p) 배로 스케일링.
        eval 모드: x를 변경 없이 반환.
        
        """
        # TODO: nn.Dropout을 사용하지 않고 구현
        raise NotImplementedError
```

## 정규화 (Normalization)

정규화 레이어는 활성화 통계량을 제어하여 학습을 안정화하는 데 도움을 줍니다.

### LayerNorm
LayerNorm은 각 샘플의 **특징 차원(Feature dimension)**(마지막 차원)에 대해 정규화를 수행합니다:

- 마지막 차원에 대해 평균(Mean)과 분산(Variance)을 계산합니다.
- 정규화: `(x - mean) / sqrt(var + eps)`
- 특징별로 학습 가능한 스케일(Scale) 및 시프트(Shift) (`weight`, `bias`)를 적용합니다.

**이 과제에서는 `elementwise_affine=True`라고 가정합니다 (항상 `weight`와 `bias`를 포함합니다).**  
`weight`와 `bias`는 각각 `(D,)` 형태를 가집니다.

LayerNorm은 배치 통계량에 의존하지 않기 때문에 트랜스포머(Transformer)에서 널리 사용됩니다.

### RMSNorm
RMSNorm은 LayerNorm과 유사하지만 제곱평균제곱근(Root-Mean-Square)만을 사용하여 정규화합니다:
- 마지막 차원에 대해 `x / sqrt(mean(x^2) + eps)`를 계산합니다.
- 일반적으로 학습 가능한 스케일(`weight`)을 포함합니다.
- 평균 차감(Mean subtraction)을 수행하지 않습니다.

RMSNorm은 속도가 더 빠르기 때문에 최신 대규모 언어 모델(LLM)에서 인기가 많습니다.

```python
class LayerNorm(nn.Module):
    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super().__init__()
        # TODO: 구현하기
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        마지막 차원에 대해 정규화합니다.
        x: (..., D)
        
        """
        # TODO: 구현하기
        raise NotImplementedError
```

```python
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-8):
        super().__init__()
        # TODO: 구현하기
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm: 마지막 차원에 대해
        x / sqrt(mean(x^2) + eps) * weight 를 적용합니다.
        
        """
        # TODO: 구현하기
        raise NotImplementedError
```

## MLP 및 잔차 네트워크 (MLPs and residual networks)

이제 레이어들을 조합하여 더 큰 네트워크를 구축합니다.

### MLP
MLP는 `depth` 개수만큼의 Linear 레이어와 그 사이에 비선형 활성화 함수(GELU 사용)를 쌓아 올린 구조입니다.
이 과제에서는 다음을 지원해야 합니다:
- 설정 가능한 깊이(Depth)
- 은닉 차원(Hidden dimension)
- 레이어 사이의 선택적인 LayerNorm (일반적인 안정화 기법)

핵심 기술은 형상(Shape)을 일관되게 유지하면서 `nn.ModuleList` / `nn.Sequential`을 사용하여 네트워크를 구축하는 것입니다.

### 트랜스포머 스타일 피드포워드 (Transformer-style FeedForward, FFN)
트랜스포머 블록은 위치별 피드포워드 네트워크(Position-wise feedforward network)를 포함합니다:
- `D -> 4D -> D` (기본값)
- 활성화 함수는 일반적으로 **GELU**를 사용합니다.

이것은 본질적으로 각 토큰 위치에 독립적으로 적용되는 MLP입니다.

### 잔차 연결 래퍼 (Residual wrapper)
잔차 연결(Residual connection)은 가장 단순한 형태의 "지름길 연결(Skip connection)"입니다:
- 출력은 `x + fn(x)`입니다.

이것은 그래디언트 흐름(Gradient flow)을 개선하고 더 깊은 네트워크를 더 안정적으로 학습할 수 있게 해줍니다.

```python
class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        use_layernorm: bool = False,
    ):
        super().__init__()
        # TODO: 모듈 빌드 (Linear + 활성화 함수의 리스트)
        # 선택적으로 레이어 사이에 LayerNorm을 삽입합니다.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 구현
        raise NotImplementedError
```

```python
class FeedForward(nn.Module):
    """
    Transformer 스타일 FFN: D -> 4D -> D (기본값)
    
    """

    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        # TODO: 두 개의 Linear 레이어를 생성하고 활성화 함수(GELU)를 선택하세요.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 구현하세요.
        raise NotImplementedError
```

```python
class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        # TODO: 구현하기
        raise NotImplementedError

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # TODO: x + fn(x, ...) 반환
        raise NotImplementedError
```

## 분류 문제 (Classification problem)

이 섹션에서는 모든 요소를 결합하여 최소한의 MNIST 분류 실험을 수행합니다.

다음 단계를 수행하게 됩니다:
1) MNIST 데이터셋 다운로드 및 로드
2) 로짓으로부터 교차 엔트로피 구현 (log-softmax를 사용하여 안정적으로 구현)
3) 간단한 MLP 기반 분류기 구축 (먼저 MNIST 이미지를 평탄화(Flatten)해야 함)
4) 최소한의 학습 루프 작성
5) 학습 손실(Train loss) 곡선 및 최종 정확도(Accuracy) 보고

여기서 목표는 최첨단(State-of-the-art) 정확도에 도달하는 것이 아니라, 전체 파이프라인을 이해하는 것입니다:
데이터 → 모델 → 로짓 → 손실 → 그래디언트 → 매개변수 업데이트.

### 모델 참고 사항
- 위에서 구현한 MLP와 아래에서 정의하는 분류 헤드(Classification head)를 하나의 모델로 결합해야 합니다.

### MNIST 참고 사항
- MNIST 이미지는 `28×28` 크기의 그레이스케일(Grayscale) 이미지입니다.
- `ToTensor()`를 거친 후, 각 이미지는 `(1, 28, 28)` 형태를 가지며 값은 `[0, 1]` 범위에 있습니다.
- MLP 분류기를 위해, 이를 길이 `784`인 벡터로 평탄화합니다.

### 자기 점검
- 최종 정확도와 함께 학습 손실 곡선 그래프를 확인하십시오.
- 모델이 올바르게 구현되었는지 확인하기 위해 최소 70% 이상의 정확도를 목표로 삼아 보세요.

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

```python
transform = transforms.ToTensor()  # -> [0,1] 범위의 float32, 형상 (1, 28, 28)

train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# TODO: 데이터로더 정의
```

```python
def cross_entropy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    logits로부터 평균 크로스 엔트로피 손실(cross-entropy loss)을 계산합니다.

    logits: (B, C)
    targets: (B,) int64

    요구사항:
    - 수치적 안정성을 이해하기 위해 log-softmax를 직접 사용하세요 (torch.nn.CrossEntropyLoss는 사용하지 마세요).
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

```python
class ClassificationHead(nn.Module):
    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        # TODO: 구현하기
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_in)
        반환값: (..., num_classes) 크기의 로짓(logits)
        
        """
        # TODO: 구현하기
        raise NotImplementedError
```

```python
def accuracy(loader):
    # TODO: 이 함수를 사용하여 모델의 정확도를 평가할 수 있습니다.
    raise NotImplementedError
```

```python
def train_classifier(
    model: nn.Module,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    lr: float,
    epochs: int,
    seed: int = 0,
) -> list[float]:
    """
    MNIST 분류를 위한 최소한의 학습 루프.

    단계:
    - 옵티마이저 정의
    - 각 에포크(epoch)마다:
        - 미니배치 샘플링
        - forward -> cross-entropy -> backward -> 옵티마이저 step
      - 각 에포크가 끝날 때 테스트 정확도 계산
    - 학습 손실(loss) 리스트 반환 (업데이트 단계당 1개)

    요구사항:
    - 학습 중에는 model.train()을, 평가 중에는 model.eval()을 호출할 것
    - torch.nn.CrossEntropyLoss를 사용하지 말 것 (직접 작성한 cross_entropy_from_logits를 사용)
    """
    # TODO: 구현하기
    raise NotImplementedError
```

