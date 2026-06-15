"""SO-100 원격 제어 모방 학습을 위한 데이터셋 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Normalizer:
    """상태(states) 및 행동(actions)을 위한 피처별 정규화기(normalizer)."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.maximum(std, eps)

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean


def _parse_key_spec(spec: str) -> tuple[str, slice]:
    """``"state_cube[:3]"``과 같은 키 사양을 (key, col_slice) 형태로 파싱합니다.

    슬라이싱 표기법을 지원합니다: ``key``, ``key[:N]``, ``key[M:]``, ``key[M:N]``.
    배열 이름과 axis=1에 적용할 열 슬라이스를 반환합니다.
    
    """
    if "[" not in spec:
        return spec, slice(None)
    name, rest = spec.split("[", 1)
    rest = rest.rstrip("]")
    parts = rest.split(":")
    if len(parts) == 2:
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        return name, slice(start, stop)
    raise ValueError(
        f"Invalid key spec: {spec!r}  (expected 'key', 'key[:N]', 'key[M:]', or 'key[M:N]')"
    )


def load_zarr(
    zarr_path: Path,
    state_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """처리된 .zarr 파일에서 states, actions, episode_ends를 로드합니다.

    Args:
        zarr_path: 처리된 .zarr 저장소의 경로.
        state_keys: 상태(state)로 연결할 데이터 배열 키 사양의 리스트.
            각 항목은 선택적으로 열 슬라이스를 포함할 수 있습니다. 예: 
            ``["state_ee_xyz", "state_cube[:3]"]``.
            ``None``인 경우, zarr 메타데이터의 ``state_key`` 속성을 기본값으로 사용합니다.
        action_keys: 행동(action)으로 연결할 데이터 배열 키 사양의 리스트.
            열 슬라이싱을 지원합니다. 예: ``["action_ee_xyz", "action_gripper"]``.
            ``None``인 경우, zarr 메타데이터의 ``action_key`` 속성을 기본값으로 사용합니다.

    Returns:
        states, actions, episode_ends
    
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    data = root["data"]

    # ── states: 하나 이상의 배열을 연결(concatenate)합니다 ────────────────────────
    if state_keys is None:
        sk = root.attrs.get("state_key", "state")
        state_keys = [sk]

    state_parts: list[np.ndarray] = []
    for spec in state_keys:
        name, col_slice = _parse_key_spec(spec)
        arr = np.asarray(data[name][:], dtype=np.float32)
        state_parts.append(arr[:, col_slice] if col_slice != slice(None) else arr)
    states = (
        np.concatenate(state_parts, axis=1) if len(state_parts) > 1 else state_parts[0]
    )

    # ── actions: 하나 이상의 배열을 연결(concatenate)합니다 ───────────────────────
    if action_keys is None:
        ak = root.attrs.get("action_key", "action")
        action_keys = [ak]

    action_parts: list[np.ndarray] = []
    for spec in action_keys:
        act_name, act_slice = _parse_key_spec(spec)
        arr = np.asarray(data[act_name][:], dtype=np.float32)
        action_parts.append(arr[:, act_slice] if act_slice != slice(None) else arr)
    actions = (
        np.concatenate(action_parts, axis=1)
        if len(action_parts) > 1
        else action_parts[0]
    )

    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)

    return states, actions, episode_ends


def load_and_merge_zarrs(
    zarr_paths: list[Path],
    state_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """여러 개의 처리된 .zarr 저장소에서 데이터를 로드하고 연결합니다.

    각 zarr 저장소는 :func:`load_zarr`를 통해 독립적으로 로드되며,
    그 결과들이 연결됩니다. 에피소드 종료(episode-end) 인덱스는 연결 후에도
    전체 데이터셋 기준에서 올바르게 유지되도록 시프트(shift)됩니다.

    :func:`load_zarr`와 동일한 ``(states, actions, episode_ends)`` 튜플을
    반환합니다.
    
    """
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_ep_ends: list[np.ndarray] = []
    offset = 0

    for zp in zarr_paths:
        states, actions, ep_ends = load_zarr(
            zp, state_keys=state_keys, action_keys=action_keys,
        )
        all_states.append(states)
        all_actions.append(actions)
        all_ep_ends.append(ep_ends + offset)
        offset += states.shape[0]

    merged_states = np.concatenate(all_states, axis=0)
    merged_actions = np.concatenate(all_actions, axis=0)
    merged_ep_ends = np.concatenate(all_ep_ends, axis=0)

    return merged_states, merged_actions, merged_ep_ends


def build_valid_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    """길이가 ``chunk_size``인 전체 행동 청크가 들어맞는 평탄화된(flat) 인덱스들을 반환합니다.

    각 에피소드 [start, end)에 대해 start … (end - chunk_size) 범위의 인덱스를 유지합니다.
    
    """
    starts = np.concatenate(([0], episode_ends[:-1]))
    indices: list[int] = []
    for start, end in zip(starts, episode_ends, strict=True):
        last_start = end - chunk_size
        if last_start < start:
            continue
        indices.extend(range(start, last_start + 1))
    return np.asarray(indices, dtype=np.int64)


class SO100ChunkDataset(Dataset):
    """크기가 H인 슬라이딩 윈도우를 적용한 (state, action_chunk) 쌍의 데이터셋.

    각 샘플은 다음으로 구성됩니다:
        state:        (state_dim,)             - 타임스텝 t에서의 상태
        action_chunk: (chunk_size, action_dim) - 행동들 [t, t+1, …, t+H-1]
    
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        chunk_size: int,
        normalizer: Normalizer | None = None,
    ) -> None:
        self.states = states
        self.actions = actions
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        self.indices = build_valid_indices(episode_ends, chunk_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(self.indices[idx])
        state = self.states[t]
        action_chunk = self.actions[t : t + self.chunk_size]

        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
            action_chunk = self.normalizer.normalize_action(action_chunk)

        state_t = torch.from_numpy(state).float()
        action_t = torch.from_numpy(action_chunk).float()

        return state_t, action_t
