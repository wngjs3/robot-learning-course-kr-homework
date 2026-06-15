"""원격 조작(teleop) 녹화 및 DAgger 데이터 수집을 위한 공유 유틸리티.

제공 기능:
- ``ZarrEpisodeWriter`` — 상태/액션 데이터를 위한 점진적(incremental) zarr 기록기.
- ``rotate_quaternion`` — 쿼터니언 회전 헬퍼 함수.
- ``load_keymap`` — ``keymap.json``을 ``{raw_keycode: action_name}`` 딕셔너리로 로드.
- ``handle_teleop_key`` — 단일 원격 조작 이동 액션을 시뮬레이션에 적용.
- ``compose_camera_views`` — 렌더링된 카메라 이미지를 2행 레이아웃으로 배치.
- 공통 상수 (``JOINT_NAMES``, ``CAMERA_NAMES`` 등)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pyquaternion as pyq
import zarr

# ── 상수 (constants) ──────────────────────────────────────────────────

JOINT_NAMES: tuple[str, ...] = (
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
)
CAMERA_NAMES: tuple[str, ...] = ("left_wrist", "angle", "top")

DEFAULT_KEYMAP_PATH: Path = Path(__file__).resolve().parent / "keymap.json"

CUBE_JOINT_NAME: str = "red_box_joint"
CUBE_DIM: int = 7  # 자유 관절(free joint): pos(3) + quat_wxyz(4)
OBSTACLE_DIM: int = 3  # 장애물 xyz 위치


# ── 쿼터니언 회전 (quaternion rotation) ───────────────────────────────


def rotate_quaternion(
    quat_wxyz: np.ndarray, axis_xyz, angle_deg: float
) -> np.ndarray:
    """*quat_wxyz*를 *axis_xyz* 축 기준으로 *angle_deg* 도만큼 회전합니다."""
    angle_rad = np.deg2rad(angle_deg)
    axis_xyz = np.asarray(axis_xyz, dtype=np.float64)
    axis_xyz = axis_xyz / np.linalg.norm(axis_xyz)
    q = pyq.Quaternion(quat_wxyz) * pyq.Quaternion(axis=axis_xyz, angle=angle_rad)
    q = q.normalised
    return q.elements  # wxyz


# ── 키맵 로드 (keymap loading) ────────────────────────────────────────


def load_keymap(km_path: Path | None = None) -> dict[int, str]:
    """``keymap.json`` 파일을 로드하여 ``{raw_keycode: action_name}``을 반환합니다."""
    km_path = km_path or DEFAULT_KEYMAP_PATH
    if not km_path.exists():
        raise FileNotFoundError(
            f"Key mapping file not found: {km_path}\n"
            "Please run  python scripts/configure_keys.py  first."
        )
    with open(km_path) as f:
        km_data = json.load(f)
    return {int(entry["raw"]): action_name for action_name, entry in km_data.items()}


# ── 원격 조작 키 처리 (teleop key dispatch) ───────────────────────────


def handle_teleop_key(
    action_name: str,
    data: mujoco.MjData,
    model: mujoco.MjModel,
    mocap_id: int,
    jaw_act_idx: int,
) -> None:
    """단일 원격 조작 이동 액션을 MuJoCo 시뮬레이션에 적용합니다.

    Parameters
    ----------
    action_name : str
        액션 식별자 (``"move_up"``, ``"rot_x_pos"``, ``"gripper_open"``, …).
    data : mujoco.MjData
        활성화된 시뮬레이션 데이터 (내부에서 직접 수정됨).
    model : mujoco.MjModel
        MuJoCo 모델 (ctrl 범위 제한에 사용됨).
    mocap_id : int
        ``data.mocap_pos`` / ``data.mocap_quat``의 인덱스.
    jaw_act_idx : int
        조(jaw)/그리퍼의 액추에이터 인덱스.
    
    """
    if action_name == "move_up":
        data.mocap_pos[mocap_id, 2] += 0.01
    elif action_name == "move_down":
        data.mocap_pos[mocap_id, 2] -= 0.01
    elif action_name == "move_left":
        data.mocap_pos[mocap_id, 0] -= 0.01
    elif action_name == "move_right":
        data.mocap_pos[mocap_id, 0] += 0.01
    elif action_name == "move_forward":
        data.mocap_pos[mocap_id, 1] += 0.01
    elif action_name == "move_backward":
        data.mocap_pos[mocap_id, 1] -= 0.01
    elif action_name == "rot_x_pos":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [1, 0, 0], 10
        )
    elif action_name == "rot_x_neg":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [1, 0, 0], -10
        )
    elif action_name == "rot_y_pos":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [0, 1, 0], 10
        )
    elif action_name == "rot_y_neg":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [0, 1, 0], -10
        )
    elif action_name == "rot_z_pos":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [0, 0, 1], 10
        )
    elif action_name == "rot_z_neg":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [0, 0, 1], -10
        )
    elif action_name == "gripper_open":
        data.ctrl[jaw_act_idx] += 0.10
        lo = model.actuator_ctrlrange[:, 0]
        hi = model.actuator_ctrlrange[:, 1]
        data.ctrl[:] = np.clip(data.ctrl, lo, hi)
    elif action_name == "gripper_close":
        data.ctrl[jaw_act_idx] -= 0.10
        lo = model.actuator_ctrlrange[:, 0]
        hi = model.actuator_ctrlrange[:, 1]
        data.ctrl[:] = np.clip(data.ctrl, lo, hi)


# ── 카메라 뷰 합성 (camera view composition) ──────────────────────────


def compose_camera_views(
    images: dict[str, np.ndarray],
    camera_names: tuple[str, ...] = CAMERA_NAMES,
) -> np.ndarray:
    """렌더링된 카메라 이미지를 2행 레이아웃으로 배치합니다.

    첫 번째 행 = 첫 두 카메라를 나란히 배치, 두 번째 행 = 세 번째 카메라 (패딩 처리됨).
    각 이미지에는 카메라 이름으로 라벨이 지정됩니다.

    Parameters
    ----------
    images : dict[str, np.ndarray]
        카메라 이름 → BGR uint8 이미지 매핑.
    camera_names : tuple[str, ...]
        카메라 순서 (처음 두 개는 첫 행에, 나머지는 두 번째 행에 배치).
    
    """
    views = []
    for cam in camera_names:
        img = images[cam].copy()
        cv2.putText(
            img, cam, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        views.append(img)

    top_row = np.concatenate(views[:2], axis=1)
    bottom = views[2]
    pad_w = top_row.shape[1] - bottom.shape[1]
    if pad_w > 0:
        padding = np.zeros((bottom.shape[0], pad_w, 3), dtype=bottom.dtype)
        bottom_row = np.concatenate([bottom, padding], axis=1)
    else:
        bottom_row = bottom
    return np.concatenate([top_row, bottom_row], axis=0)


# ── ZarrEpisodeWriter ────────────────────────────────────────────────


@dataclass
class ZarrEpisodeWriter:
    """원격 조작 / DAgger 상태 및 액션 데이터를 위한 점진적 zarr 기록기.

    입력되는 타임스텝 데이터를 버퍼링하고 매 *flush_every* 단계마다 디스크로 플러시합니다.
    에피소드를 완료하는 ``end_episode()`` 및 마지막으로 완료된 에피소드 이후의 모든 데이터를
    롤백하는 ``discard_episode()``를 지원합니다.
    
    """

    path: Path
    joint_dim: int = 6
    ee_dim: int = 7
    cube_dim: int = 7
    gripper_dim: int = 1
    obstacle_dim: int = 3
    flush_every: int = 12

    # ── 내부 상태 (__post_init__에 의해 채워짐) ───────────────────────────
    root: zarr.Group = field(init=False, repr=False)
    state_joints_arr: zarr.Array = field(init=False, repr=False)
    state_ee_arr: zarr.Array = field(init=False, repr=False)
    state_cube_arr: zarr.Array | None = field(init=False, repr=False, default=None)
    state_gripper_arr: zarr.Array = field(init=False, repr=False)
    action_gripper_arr: zarr.Array = field(init=False, repr=False)
    state_obstacle_arr: zarr.Array = field(init=False, repr=False)
    ep_ends_arr: zarr.Array = field(init=False, repr=False)

    _state_joints_buf: list[np.ndarray] = field(init=False, repr=False)
    _state_ee_buf: list[np.ndarray] = field(init=False, repr=False)
    _state_cube_buf: list[np.ndarray] = field(init=False, repr=False)
    _state_gripper_buf: list[np.ndarray] = field(init=False, repr=False)
    _action_gripper_buf: list[np.ndarray] = field(init=False, repr=False)
    _state_obstacle_buf: list[np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open_group(str(self.path), mode="w", zarr_format=3)

        compressor = zarr.codecs.Blosc(cname="zstd", clevel=3, shuffle=2)
        compressors = (compressor,)

        data = self.root.require_group("data")
        meta = self.root.require_group("meta")

        self.state_joints_arr = data.require_array(
            "state_joints",
            shape=(0, self.joint_dim),
            chunks=(min(self.flush_every, 4096), self.joint_dim),
            dtype="f4",
            compressors=compressors,
        )
        self.state_ee_arr = data.require_array(
            "state_ee",
            shape=(0, self.ee_dim),
            chunks=(min(self.flush_every, 4096), self.ee_dim),
            dtype="f4",
            compressors=compressors,
        )
        if self.cube_dim > 0:
            self.state_cube_arr = data.require_array(
                "state_cube",
                shape=(0, self.cube_dim),
                chunks=(min(self.flush_every, 4096), self.cube_dim),
                dtype="f4",
                compressors=compressors,
            )
        self.state_gripper_arr = data.require_array(
            "state_gripper",
            shape=(0, self.gripper_dim),
            chunks=(min(self.flush_every, 4096), self.gripper_dim),
            dtype="f4",
            compressors=compressors,
        )
        self.action_gripper_arr = data.require_array(
            "action_gripper",
            shape=(0, self.gripper_dim),
            chunks=(min(self.flush_every, 4096), self.gripper_dim),
            dtype="f4",
            compressors=compressors,
        )
        self.state_obstacle_arr = data.require_array(
            "state_obstacle",
            shape=(0, self.obstacle_dim),
            chunks=(min(self.flush_every, 4096), self.obstacle_dim),
            dtype="f4",
            compressors=compressors,
        )
        self.ep_ends_arr = meta.require_array(
            "episode_ends",
            shape=(0,),
            chunks=(1024,),
            dtype="i8",
            compressors=compressors,
        )

        self._state_joints_buf = []
        self._state_ee_buf = []
        self._state_cube_buf = []
        self._state_gripper_buf = []
        self._action_gripper_buf = []
        self._state_obstacle_buf = []

    # ── 편의용 속성 (convenience attributes) ──────────────────────────────

    def set_attrs(self, **attrs) -> None:
        """zarr 루트 그룹에 임의의 메타데이터를 저장합니다."""
        for k, v in attrs.items():
            self.root.attrs[k] = v

    @property
    def num_steps_total(self) -> int:
        """기록된 총 타임스텝 수 (플러시되지 않은 버퍼 포함)."""
        return int(self.state_joints_arr.shape[0]) + len(self._state_joints_buf)

    @property
    def num_episodes(self) -> int:
        return int(self.ep_ends_arr.shape[0])

    # ── 추가 / 플러시 / 에피소드 관리 ─────────────────────────────────────

    def append(
        self,
        state_joints: np.ndarray,
        state_ee: np.ndarray,
        state_cube: np.ndarray,
        state_gripper: np.ndarray,
        action_gripper: np.ndarray,
        state_obstacle: np.ndarray,
    ) -> None:
        """한 타임스텝의 데이터를 버퍼링합니다. 매 *flush_every* 단계마다 자동으로 플러시됩니다."""
        self._state_joints_buf.append(state_joints.astype(np.float32, copy=False))
        self._state_ee_buf.append(state_ee.astype(np.float32, copy=False))
        if self.cube_dim > 0:
            self._state_cube_buf.append(state_cube.astype(np.float32, copy=False))
        self._state_gripper_buf.append(state_gripper.astype(np.float32, copy=False))
        self._action_gripper_buf.append(action_gripper.astype(np.float32, copy=False))
        self._state_obstacle_buf.append(state_obstacle.astype(np.float32, copy=False))

        if len(self._state_joints_buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """버퍼링된 데이터를 디스크에 씁니다."""
        if not self._state_joints_buf:
            return

        joints = np.stack(self._state_joints_buf, axis=0)
        ee = np.stack(self._state_ee_buf, axis=0)
        gripper = np.stack(self._state_gripper_buf, axis=0)
        action_grip = np.stack(self._action_gripper_buf, axis=0)
        obstacle = np.stack(self._state_obstacle_buf, axis=0)

        n0 = self.state_joints_arr.shape[0]
        n1 = n0 + joints.shape[0]

        self.state_joints_arr.resize((n1, self.joint_dim))
        self.state_ee_arr.resize((n1, self.ee_dim))
        if self.state_cube_arr is not None:
            self.state_cube_arr.resize((n1, self.cube_dim))
        self.state_gripper_arr.resize((n1, self.gripper_dim))
        self.action_gripper_arr.resize((n1, self.gripper_dim))
        self.state_obstacle_arr.resize((n1, self.obstacle_dim))
        self.state_joints_arr[n0:n1] = joints
        self.state_ee_arr[n0:n1] = ee
        if self.state_cube_arr is not None:
            cube = np.stack(self._state_cube_buf, axis=0)
            self.state_cube_arr[n0:n1] = cube
        self.state_gripper_arr[n0:n1] = gripper
        self.action_gripper_arr[n0:n1] = action_grip
        self.state_obstacle_arr[n0:n1] = obstacle

        self._state_joints_buf.clear()
        self._state_ee_buf.clear()
        self._state_cube_buf.clear()
        self._state_gripper_buf.clear()
        self._action_gripper_buf.clear()
        self._state_obstacle_buf.clear()

    def end_episode(self) -> None:
        """플러시를 수행하고 에피소드 경계를 기록합니다."""
        self.flush()
        end_idx = int(self.state_joints_arr.shape[0])
        m0 = self.ep_ends_arr.shape[0]
        self.ep_ends_arr.resize((m0 + 1,))
        self.ep_ends_arr[m0] = end_idx

    def discard_episode(self) -> None:
        """마지막 ``end_episode()`` 호출 이후 기록된 모든 데이터를 롤백합니다."""
        self._state_joints_buf.clear()
        self._state_ee_buf.clear()
        self._state_cube_buf.clear()
        self._state_gripper_buf.clear()
        self._action_gripper_buf.clear()
        self._state_obstacle_buf.clear()

        if self.ep_ends_arr.shape[0] > 0:
            rollback_to = int(self.ep_ends_arr[-1])
        else:
            rollback_to = 0

        if self.state_joints_arr.shape[0] > rollback_to:
            self.state_joints_arr.resize((rollback_to, self.joint_dim))
            self.state_ee_arr.resize((rollback_to, self.ee_dim))
            if self.state_cube_arr is not None:
                self.state_cube_arr.resize((rollback_to, self.cube_dim))
            self.state_gripper_arr.resize((rollback_to, self.gripper_dim))
            self.action_gripper_arr.resize((rollback_to, self.gripper_dim))
            self.state_obstacle_arr.resize((rollback_to, self.obstacle_dim))
