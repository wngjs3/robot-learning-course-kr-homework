# ex2 보기용 문서

> 이 파일은 GitHub에서 안정적으로 읽기 위한 Markdown 버전입니다. 직접 실행하려면 [`src/ex2.ipynb`](../src/ex2.ipynb) 노트북을 사용하세요.

# 과제 2: PyTorch 핵심 (PyTorch core)

이 과제에서는 작성하는 거의 모든 모델에서 재사용하게 될 PyTorch의 핵심 "근육 기억(muscle memory)"을 기르게 됩니다:

- **자동 미분(Autograd)**: 그래디언트(gradient)가 생성되고 누적되는 방식, 그리고 하나 또는 여러 입력에 대한 그래디언트를 계산하는 방법.
- **데이터 로딩(Dataloading)**: 소규모 `Dataset` 작성, `DataLoader` 사용 및 사용자 정의 `collate_fn` 구현.
- **옵티마이저(Optimizers)**: **AdamW** 업데이트를 처음부터 직접 구현(상태(state), 편향 수정(bias correction), 가중치 감쇠(weight decay)).
- **훈련 기초(Training basics)**: 깔끔한 단일 훈련 단계(training step) 구현.
- **초기화(Initialization)**: 팬인/팬아웃(fan-in/out) 및 일반적인 초기화 방법(Xavier / Kaiming), 그리고 `nn.Linear`를 초기화하는 헬퍼 함수.

이전과 마찬가지로, 함수 이름이나 시그니처를 변경하지 않고 모든 `TODO`를 채워 넣으세요.
디버깅할 때는 크기(shape)/데이터 타입(dtype)/디바이스(device)를 출력하고, 간단한 정상성 검사(sanity check)를 작성하여 비교해 보세요(예: PyTorch의 내장 기능과 비교).

```python
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
```

## 자동 미분(Autograd)의 기초

PyTorch는 `requires_grad=True`인 텐서에 연산을 적용할 때 계산 그래프(computation graph)를 구축합니다.
`backward()`(또는 `torch.autograd.grad`)를 호출하면 해당 그래프를 거슬러 올라가며 그래디언트를 계산합니다.

### 핵심 개념
- **리프 텐서(Leaf tensor)**: 연산의 결과가 아니라 사용자가 직접 생성한 `requires_grad=True`인 텐서입니다. 리프 텐서는 그래디언트를 `.grad`에 저장할 수 있습니다.
- **그래디언트 누적(Gradient accumulation)**: `backward()`를 호출하면 `.grad`에 값이 더해집니다(덮어쓰지 않습니다). 따라서 단계/호출 사이에 그래디언트를 재설정(reset)해야 합니다.
- **`torch.autograd.grad` vs `.backward()`**
  - `torch.autograd.grad(f, x)`는 `df/dx`를 직접 반환하며, 명시적으로 지정하지 않는 한 `x.grad`에 값을 기록하지 않습니다.
  - `f.backward()`는 리프 텐서의 `.grad`에 그래디언트를 기록합니다.

다음 함수들에서는 두 API를 모두 사용하여 `f(x) = sum(x^2)`과 같은 간단한 스칼라 함수의 그래디언트를 계산합니다.

### `torch.no_grad()`
추론(inference) 전용 코드에서는 그래디언트 추적 및 그래프 구축을 방지하기 위해 이 데코레이터/컨텍스트 매니저로 감쌉니다:
- 메모리를 절약합니다.
- 평가 속도를 높입니다.

### `detach()`
`y = x.detach()`는 `x`와 데이터를 공유하지만 자동 미분 그래프와는 **연결되지 않은** 텐서를 반환합니다.
이는 특정 값을 상수의 타겟(target)으로 취급하고 싶을 때 유용합니다.

### `model.train()` vs `model.eval()`
- `train()`은 훈련 동작을 활성화합니다(예: 드롭아웃(dropout) 활성화, 배치 정규화(batchnorm)가 이동 평균 통계량을 업데이트함).
- `eval()`은 추론 동작을 활성화합니다(예: 드롭아웃 비활성화, 배치 정규화가 이동 평균 통계량을 사용함).

```python
def grad_with_autograd_grad(x: torch.Tensor) -> torch.Tensor:
    """
    torch.autograd.grad를 사용하여 f(x) = sum(x^2)의 그래디언트를 계산합니다.

    요구사항:
    - .backward()를 호출하지 마세요.
    - 함수 내부에서 x는 그래디언트가 필요해야 합니다(이미 필요하다고 가정하지 마세요).
    - 반드시 df/dx를 반환해야 합니다.
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

```python

def grad_with_backward(x: torch.Tensor) -> torch.Tensor:
    """
    .backward()를 사용하여 f(x) = sum(x^2)의 그래디언트를 계산합니다.

    요구사항:
    - 반드시 df/dx를 반환해야 함
    - 호출 간에 그래디언트가 누수되지 않아야 함 (x.grad 누적 주의)
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

```python
def grad_wrt_multiple_inputs(
    a: torch.Tensor, b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    여러 입력에 대한 그래디언트를 계산합니다. 함수는 f(a, b) = sum(a^2 + ab)입니다.

    반환값:
        (df/da, df/db)

    요구사항:
    - torch.autograd.grad를 사용할 것
    - 이 함수 내에서 a와 b 모두가 grad를 필요로 하도록(require grad) 설정할 것
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

## 데이터 로딩 (Dataloading)

PyTorch에서 `Dataset`은 *단일* 훈련 샘플을 가져오는 방법을 정의하고, `DataLoader`는 다음 작업을 처리합니다:
- 배치화(batching)
- 셔플링(shuffling)
- 병렬 워커(parallel workers)
- `collate_fn`을 통한 선택적인 사용자 정의 배치화 로직

### `Dataset` 한 줄 요약
`Dataset`은 오직 다음 두 가지만 필요로 합니다:
- `__len__`: 아이템의 개수
- `__getitem__`: 하나의 아이템 반환 (예: `(x, y)`)

### `collate_fn`이 중요한 이유
기본 DataLoader의 콜레이션(collation)은 아이템들을 새로운 배치 차원을 따라 쌓아 올립니다(stack).
이는 고정된 크기의 텐서에는 잘 작동하지만, **가변 길이 시퀀스(variable-length sequences)**에서는 작동하지 않고 에러가 발생합니다.

따라서 직접 패딩(padding)을 구현해 보겠습니다:
- 1D 토큰 시퀀스의 리스트를 패딩 처리된 텐서 `(B, T_max)`로 변환합니다.
- 실제 길이(`lengths`)와 패딩 마스크(`padding_mask`)를 추적합니다.

### 패딩 마스크 규칙
본 과제의 패딩 마스크 규칙은 다음과 같습니다:
- `padding_mask[b, t] == True`는 **패딩 영역 / 유효하지 않은 토큰**을 의미합니다.
- `padding_mask[b, t] == False`는 **실제 토큰**을 의미합니다.

```python
from torch.utils.data import DataLoader, Dataset
```

```python
class TensorPairDataset(Dataset):
    """
    (x, y)를 감싸는 최소한의 데이터셋.

    x: (N, ...)
    y: (N, ...)

    N은 샘플 수입니다. 데이터셋은 (x[i], y[i]) 튜플을 반환해야 합니다.
    
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        # TODO: 구현하기
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO: 구현하기
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: 구현하기
        raise NotImplementedError
```

```python
class NextTokenDataset(Dataset):
    """
    다음 토큰 예측(Next-token prediction) 데이터셋.

    (N, T) 크기의 토큰이 주어지면 다음을 생성합니다:
      input_ids  = tokens[:, :-1]
      target_ids = tokens[:, 1:]

    아이템별 반환 값:
      (input_ids, target_ids)

    Notes:
    - 반환되는 텐서는 길이가 (T-1)인 1D 텐서여야 합니다.
    - dtype은 정수형(integer)을 유지해야 합니다.
    
    """

    def __init__(self, tokens: torch.Tensor):
        # TODO: 구현하기
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO: 구현하기
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: 구현하기
        raise NotImplementedError
```

```python

class RandomCropSequenceDataset(Dataset):
    """
    고정된 길이의 무작위 크롭(crop)을 반환하는 시퀀스 데이터셋.

    tokens: (N, T_total)
    crop_len: L

    각 __getitem__ 호출 시:
      - s+L <= T_total을 만족하는 시작 인덱스 s를 샘플링
      - tokens[idx, s:s+L]을 반환

    요구사항:
    - seed가 제공되는 경우 결정론적 동작을 위해 torch.Generator를 사용하세요.
    - Python의 random 모듈은 사용하지 마세요.
    
    """

    def __init__(self, tokens: torch.Tensor, crop_len: int, seed: int | None = None):
        # TODO: 구현하기
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO: 구현하기
        raise NotImplementedError

    def __getitem__(self, idx: int) -> torch.Tensor:
        # TODO: 구현하기
        raise NotImplementedError
```

```python


@dataclass(frozen=True)
class PaddedBatch:
    """
    가변 길이 시퀀스를 위한 패딩된 배치.

    tokens: LongTensor (B, T_max)
    lengths: LongTensor (B,)
    padding_mask: BoolTensor (B, T_max), True는 "패딩임"을 의미함
    
    """

    tokens: torch.Tensor
    lengths: torch.Tensor
    padding_mask: torch.Tensor


def pad_1d_sequences(seqs: list[torch.Tensor], pad_value: int = 0) -> PaddedBatch:
    """
    1D 정수 텐서 리스트를 동일한 길이로 패딩합니다.

    요구사항:
    - PaddedBatch(tokens, lengths, padding_mask) 반환
    - padding_mask[b, t] == True는 t >= lengths[b]일 때만 성립
    - tokens는 dtype이 long이어야 하며, 그렇지 않으면 캐스팅할 것
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

```python
def collate_next_token_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_value: int = 0
) -> dict[str, torch.Tensor]:
    """
    가변 길이를 가질 수 있는 NextTokenDataset 샘플들을 위한 collate 함수.

    batch: (input_ids, target_ids)의 리스트, 각 원소는 1D

    반환하는 dict 구성:
      - input_ids: (B, T_max)
      - target_ids: (B, T_max)
      - attention_mask: (B, T_max), True는 "유지"를 의미 (패딩이 아님)
      - padding_mask: (B, T_max), True는 "패딩"을 의미

    요구사항:
    - input_ids와 target_ids를 일관되게 패딩할 것
    - attention_mask는 padding_mask의 논리적 NOT 연산 결과여야 함
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

```python
def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    collate_fn=None,
    num_workers: int = 0,
) -> DataLoader:
    """
    선택적 collate_fn을 사용하여 DataLoader를 생성합니다.
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

## 옵티마이저 (AdamW 직접 구현)

PyTorch 옵티마이저는 각 매개변수(parameter)에 대한 **상태(state)**를 유지합니다(예: Adam의 모멘트 추정치).
이 섹션에서는 Adam에 *분리된(decoupled)* 가중치 감쇠가 추가된 **AdamW**를 직접 구현합니다.

### AdamW 상태
각 매개변수 텐서 `p`에 대해 다음을 저장합니다:
- `m`: 1차 모멘트 (그래디언트의 지수 이동 평균(EMA))
- `v`: 2차 모멘트 (그래디언트 제곱의 지수 이동 평균(EMA))
- `t`: 단계(step) 카운터

### 업데이트 개요 (하이 레벨)
1) 모멘트 `m, v` 업데이트
2) 편향 수정 적용 (`m_hat, v_hat`)
3) 매개변수 업데이트 적용:
   `p -= lr * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * p )`

참고:
- 이 업데이트는 **인플레이스(in-place)** 연산으로 수행됩니다(`p`를 직접 수정).
- 그래디언트는 수정되어서는 안 됩니다.
- 상태 텐서들은 매개변수의 크기(shape)/디바이스(device)/데이터 타입(dtype)과 일치해야 합니다.

```python
@dataclass
class AdamWState:
    """
    파라미터별 AdamW 상태.

    m: 1차 모멘트
    v: 2차 모멘트
    t: 스텝 수
    
    """

    m: torch.Tensor
    v: torch.Tensor
    t: int


def init_adamw_state(p: torch.Tensor) -> AdamWState:
    """
    파라미터 텐서 p에 대한 AdamW 상태 텐서를 초기화합니다.

    생성할 항목:
    - m: p와 동일한 형상/디바이스/dtype을 가진 zero 텐서
    - v: p와 동일한 형상/디바이스/dtype을 가진 zero 텐서
    - t: 0에서 시작하는 스텝 카운터

    참고 사항 / 요구 사항:
    - m과 v에는 torch.zeros_like(p)를 사용하세요.
    - 상태에 그래디언트를 부여하지 마세요 (torch.no_grad() 하에서 초기화).
    - t는 0에서 시작합니다. adamw_step_에서 편향 보정 항 (1 - beta1^t) 및 (1 - beta2^t)를
      계산하기 *전에* 첫 번째 업데이트 시 t를 1로 증가시키세요.
    - 상태 텐서는 p와 동일한 디바이스(CPU 또는 GPU)에 있어야 하며,
      p와 동일한 dtype을 가져야 합니다.
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

```python
def adamw_step_(
    p: torch.Tensor,
    grad: torch.Tensor,
    state: AdamWState,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> AdamWState:
    """
    In-place AdamW 매개변수 업데이트 (p를 업데이트함).

    알고리즘 (AdamW):
      m = beta1*m + (1-beta1)*grad
      v = beta2*v + (1-beta2)*grad^2
      m_hat = m / (1 - beta1^t)
      v_hat = v / (1 - beta2^t)
      p = p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)

    요구사항:
    - p를 in-place로 업데이트할 것.
    - 업데이트된 state를 반환할 것 (t가 증가함).
    - grad를 수정하지 말 것.
    - 어떤 텐서 형태(shape)에 대해서도 동작해야 함.
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

```python
def adamw_step_many_(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    states: list[AdamWState],
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> list[AdamWState]:
    """
    여러 파라미터에 AdamW를 적용합니다.

    요구사항:
    - len(params) == len(grads) == len(states)
    - 각 파라미터를 in-place로 업데이트합니다.
    - 업데이트된 states의 리스트를 반환합니다.
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

## 훈련 기초 (Training basics)

최소한의 훈련 단계는 거의 모든 곳에서 동일한 패턴을 따릅니다:

1) 모델을 훈련 모드로 설정
2) 그래디언트 초기화
3) 순전파(forward pass) 진행
4) 손실(loss) 계산
5) 역전파(backward pass) 진행
6) 옵티마이저 단계(step) 진행

이 연습에서는 표준 PyTorch 옵티마이저를 사용하여 단일 MSE 훈련 단계를 구현합니다.
반환 값은 Python float 타입의 손실 값이어야 합니다.

```python
def train_step_mse(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    표준 torch 옵티마이저를 사용하는 하나의 MSE 학습 스텝.
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

## 매개변수 초기화 (Parameter initialization)

초기화는 훈련 시작 시 신호와 그래디언트의 스케일을 제어하기 때문에 매우 중요합니다.

### 팬인(Fan-in) / 팬아웃(Fan-out)
- `fan_in`: 유닛으로 들어오는 입력 연결의 수
- `fan_out`: 유닛에서 나가는 출력 연결의 수

크기가 `(out_features, in_features)`인 Linear 레이어 가중치(weight)의 경우:
- `fan_in = in_features`
- `fan_out = out_features`

### 일반적인 초기화 기법
- **Xavier / Glorot** (tanh / 선형에 가까운 네트워크에 주로 적합):
  활성화 함수가 대략 대칭적일 때 레이어 전반에 걸쳐 분산(variance)을 안정적으로 유지합니다.
- **Kaiming / He** (ReLU 계열 네트워크에 주로 적합):
  ReLU가 입력의 약 절반을 0으로 만드는 점을 감안하여 설계되었습니다.

이 섹션에서는 Xavier uniform 및 Kaiming uniform을 구현하고, 이를 사용하여 `nn.Linear`를 초기화합니다.
또한 명시적인 지시가 없는 한 편향(bias)은 항상 0으로 초기화합니다.

```python
def fan_in_fan_out(weight: torch.Tensor) -> tuple[int, int]:
    """가중치 텐서의 (fan_in, fan_out)을 계산합니다."""
    # TODO: 구현하기
    raise NotImplementedError
```

```python
def xavier_uniform_(weight: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    In-place Xavier/Glorot 균등 분포(uniform) 초기화:
      bound = gain * sqrt(6 / (fan_in + fan_out))
      U(-bound, bound)
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

```python
def kaiming_uniform_(weight: torch.Tensor, nonlinearity: str = "relu") -> torch.Tensor:
    """
    In-place Kaiming/He uniform 초기화.

    다음과 같은 일반적인 설정을 따릅니다:
      ReLU의 경우 gain = sqrt(2)
      std = gain / sqrt(fan_in)
      bound = sqrt(3) * std
      U(-bound, bound)
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

```python
def init_linear_(layer: nn.Linear, scheme: str = "xavier") -> nn.Linear:
    """
    nn.Linear를 인플레이스(in-place)로 초기화합니다.

    scheme:
      - "xavier"
      - "kaiming_relu"
      - "zero" (가중치 및 편향 = 0)
    
    """
    # TODO: 구현하기
    raise NotImplementedError
```

