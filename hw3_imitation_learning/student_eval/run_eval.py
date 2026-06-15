# !/usr/bin/env python3
"""HW3 실습 1-3을 위한 로컬 평가 스크립트.

사용법
-----
    python run_eval.py --exercise 1 --checkpoint ex1.pt
    python run_eval.py --exercise 2 --checkpoint ex2.pt
    python run_eval.py --exercise 3 --checkpoint ex3.pt

이 스크립트는 프로젝트 루트(즉, ``student_eval/``의 상위 디렉터리) 기준 ``hw3/model.py``에 ``model.py``가 있을 것으로 예상합니다.

이 스크립트는 동일한 디렉터리에 있는 **컴파일된** ``eval_harness`` 모듈(.so / .pyd)을 임포트합니다. 이를 수정하거나 대체하지 마십시오.

스크립트 수행 작업:
  1. ``./model.py``에서 모델 정의 로드
  2. 체크포인트에서 학습된 가중치 로드
  3. 100회의 헤드리스 시뮬레이션 에피소드 실행 (seed=42)
  4. 성공률 및 요약 지표 출력
  5. 서명된 ``ex{N}_result.hwresult`` 파일 생성

생성된 ``.hwresult`` 파일은 자기 점검용 결과 기록으로 보관할 수 있습니다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EX_INFO = {
    1: {"name": "Single-Cube Obstacle (train)", "default_ckpt": "ex1.pt"},
    2: {"name": "Single-Cube Obstacle (adversarial)", "default_ckpt": "ex2.pt"},
    3: {"name": "Multicube Goal-Conditioned", "default_ckpt": "ex3.pt"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HW3 – Local Evaluation (Exercises 1–3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exercise",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Exercise number to evaluate (1, 2, or 3).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to your checkpoint (default: ./ex{N}.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the signed result file "
        "(default: ./ex{N}_result.hwresult)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42). Keep fixed when comparing runs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode progress output.",
    )
    args = parser.parse_args()

    ex = args.exercise
    info = _EX_INFO[ex]

    # 과제 번호에 따른 기본값
    ckpt = args.checkpoint or info["default_ckpt"]
    output = args.output or f"ex{ex}_result.hwresult"

    # 경로 확인 – model.py는 항상 <project_root>/hw3/model.py에 위치함
    # project_root = student_eval/(이 스크립트가 있는 디렉터리)의 상위 디렉터리
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "hw3" / "model.py"
    ckpt_path = Path(ckpt).resolve()
    output_path = Path(output).resolve()

    if not model_path.exists():
        print(
            f"ERROR: model.py not found at {model_path}\n"
            "       Expected hw3/model.py relative to the project root.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # 컴파일된 하네스 임포트 (이 디렉터리의 .so / .pyd 파일)
    harness_dir = Path(__file__).resolve().parent
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))

    try:
        import eval_harness  # noqa: E402 — 컴파일된 .so
    except ImportError as e:
        print(
            "ERROR: Could not import eval_harness.\n"
            "Make sure the compiled eval_harness*.so (or .pyd on Windows)\n"
            "is in the same directory as this script.\n"
            f"\nDetails: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print()
    print("=" * 55)
    print(f"  HW3 Exercise {ex} – {info['name']}")
    print("=" * 55)
    print(f"  Model      : {model_path}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Output     : {output_path}")
    print(f"  Episodes   : {args.num_episodes}")
    print(f"  Seed       : {args.seed}")
    print("=" * 55)

    eval_harness.run_eval(
        exercise=ex,
        model_py=str(model_path),
        checkpoint=str(ckpt_path),
        output_path=str(output_path),
        num_episodes=args.num_episodes,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
