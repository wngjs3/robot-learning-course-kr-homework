"""SO-100 원격제어 레코더를 위한 대화형 키 설정.

작은 OpenCV 창을 열고 각 액션을 하나씩 안내합니다.
각 액션에 할당할 키를 누르세요 (추천 키 참고). 결과는
과제 전반에서 자동으로 로드되는 JSON 파일로 저장됩니다.

Usage:
    python scripts/configure_keys.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# 기본 출력 위치 (이 스크립트와 같은 위치)
DEFAULT_KEYMAP_PATH = Path(__file__).resolve().parent.parent / "hw3" / "keymap.json"

# 사람이 읽을 수 있는 설명과 기본값을 포함한 설정 가능한 모든 액션
# (raw_code, ascii_code) — 문서화 및 폴백(fallback)용으로 모두 사용됨.
ACTIONS: list[tuple[str, str]] = [
    ("move_up", "Move EE up (+Z) (recommended: UP ARROW)"),
    ("move_down", "Move EE down (-Z) (recommended: DOWN ARROW)"),
    ("move_left", "Move EE left (-X) (recommended: LEFT ARROW)"),
    ("move_right", "Move EE right (+X) (recommended: RIGHT ARROW)"),
    ("move_forward", "Move EE forward (+Y) (recommended: W)"),
    ("move_backward", "Move EE backward (-Y) (recommended: S)"),
    ("rot_x_pos", "Rotate EE +X axis"),
    ("rot_x_neg", "Rotate EE -X axis"),
    ("rot_y_pos", "Rotate EE +Y axis"),
    ("rot_y_neg", "Rotate EE -Y axis"),
    ("rot_z_pos", "Rotate EE +Z axis"),
    ("rot_z_neg", "Rotate EE -Z axis"),
    ("gripper_open", "Open gripper (Jaw +)"),
    ("gripper_close", "Close gripper (Jaw -)"),
    ("reset", "Reset environment (recommended: R)"),
    ("record", "Toggle recording on/off (recommended: Space)"),
    ("end_episode", "End recorded episode (recommended: Enter)"),
    ("escape", "Quit session (recommended: ESC)"),
    ("goal_cube_red", "Select RED cube as goal (multicube only)"),
    ("goal_cube_green", "Select GREEN cube as goal (multicube only)"),
    ("goal_cube_blue", "Select BLUE cube as goal (multicube only)"),
]

WINDOW_W = 640
WINDOW_H = 200


def draw_prompt(
    action_name: str, description: str, index: int, total: int
) -> np.ndarray:
    """사용자에게 키를 누르도록 요청하는 프롬프트 이미지 생성."""
    img = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    cv2.putText(
        img,
        f"Key config  ({index + 1}/{total})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
    )
    cv2.putText(
        img,
        f"Action: {description}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        f"({action_name})",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
    )
    cv2.putText(
        img,
        ">>> Press the key you want to use <<<",
        (10, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return img


def draw_assigned(action_name: str, raw: int, ascii_code: int) -> np.ndarray:
    """키가 캡처된 후 확인 메시지 표시."""
    img = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    label = chr(ascii_code) if 32 <= ascii_code <= 126 else "<special>"
    cv2.putText(
        img,
        f"Assigned: {action_name}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        f"Key: '{label}'  (raw={raw}, ascii={ascii_code})",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 255),
        2,
    )
    cv2.putText(
        img,
        "Moving to next action...",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
    )
    return img


def run_configuration(output_path: Path) -> None:
    """모든 액션을 차례대로 진행하며 대화형으로 키를 캡처."""
    cv2.namedWindow("Key Configuration", cv2.WINDOW_AUTOSIZE)

    keymap: dict[str, dict] = {}
    total = len(ACTIONS)

    for i, (action_name, description) in enumerate(ACTIONS):
        prompt = draw_prompt(action_name, description, i, total)
        cv2.imshow("Key Configuration", prompt)

        # 키 입력 대기 (키가 눌릴 때까지 블로킹)
        while True:
            k_raw = cv2.waitKeyEx(0)  # 0 = 무한 대기
            if k_raw != -1:
                break

        k_ascii = k_raw & 0xFF
        label = chr(k_ascii) if 32 <= k_ascii <= 126 else f"raw:{k_raw}"

        keymap[action_name] = {
            "raw": k_raw,
            "ascii": k_ascii,
            "label": label,
            "description": description,
        }

        print(f"  [{i + 1}/{total}] {action_name:20s} -> '{label}' (raw={k_raw})")

        # 간단한 확인 메시지
        confirm = draw_assigned(action_name, k_raw, k_ascii)
        cv2.imshow("Key Configuration", confirm)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(keymap, f, indent=2)
    print(f"\nKey mapping saved to {output_path}")
    print(
        "You can now run record_teleop_demos.py — it will load this mapping automatically."
    )


def load_keymap(path: Path | None = None) -> dict[str, int]:
    """keymap JSON을 로드하고 {action_name: raw_keycode}를 반환.

    파일이 존재하지 않으면 빈 dict를 반환 (호출자는 기본값을 사용해야 함).
    
    """
    if path is None:
        path = DEFAULT_KEYMAP_PATH
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {action: entry["raw"] for action, entry in data.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configure keyboard mapping for SO-100 teleop."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_KEYMAP_PATH,
        help=f"Output JSON file (default: {DEFAULT_KEYMAP_PATH}).",
    )
    args = parser.parse_args()
    run_configuration(args.output)
