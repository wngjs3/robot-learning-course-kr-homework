import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Python, NumPy, PyTorch의 랜덤 시드를 설정합니다.
    
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path) -> Path:
    """
    디렉터리가 존재하지 않으면 생성하고, 이를 Path 객체로 반환합니다.
    
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path