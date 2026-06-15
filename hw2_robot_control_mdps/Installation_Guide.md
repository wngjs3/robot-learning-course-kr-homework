# HW2 SO-100 Tutorial — 설치 가이드 (Linux & Windows)

## 사전 요구 사항 (Prerequisites)

- 가상 환경(Virtual Environment)(conda / uv / venv) 사용을 권장합니다. 여기서는 **venv**를 예시로 사용합니다.
- **Python 3.12** 버전에서 테스트되었으며 이를 권장합니다.
- **pip** (보통 Python과 함께 기본 제공됨).
- **Git** (저장소 다운로드용).

**Linux (Ubuntu/Debian)의 경우:**
MuJoCo의 뷰어(Viewer)를 실행하려면 OpenGL / EGL 드라이버가 필요합니다. 이미 설치되어 있을 가능성이 높지만, 아래 명령어를 통해 명시적으로 설치할 수 있습니다:
`sudo apt install libegl1-mesa-dev libgl1-mesa-dri libglvnd-dev`

**Windows의 경우:**
PowerShell에서 `python`과 `pip` 명령어를 사용할 수 있는지 확인하세요 (`python --version`으로 테스트). 사용할 수 없다면 시스템 환경 변수(PATH)에 Python을 추가해야 합니다.

---

## 1. 가상 환경 생성 및 활성화 (Create & Activate a Virtual Environment)
저장소의 루트 디렉터리에서 다음 명령어를 실행합니다:
**Linux (Bash):**
`python3 -m venv mujoco`
`source mujoco/bin/activate`

> **팁:** 새 터미널을 열 때마다 `source mujoco/bin/activate`를 실행하여 가상 환경을 다시 활성화해야 합니다.

**Windows (PowerShell):**
`python -m venv mujoco`
`.\mujoco\Scripts\Activate.ps1`

> **팁:** Windows에서 실행 정책(Execution Policy) 오류가 발생하는 경우, 다음 명령어를 실행하세요: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**CAB H56 & H57 실습실 컴퓨터에 가상 환경 설치하기:** 
실습실 컴퓨터를 사용하는 학생들을 위한 안내입니다. 개인 드라이브의 용량은 10 GB로 제한되어 있습니다. 가상 환경은 약 7-8 GB의 공간을 차지합니다. 따라서 실행 가능한 해결책은 `/tmp` 폴더(16 GB)에 venv를 설치하고, 본인의 코드는 개인 드라이브에 저장하는 것입니다. venv를 설치하는 bash 스크립트를 작성해 두면, 실습실 컴퓨터에 로그인할 때마다 스크립트를 실행하여 `/tmp`에 venv를 복구할 수 있습니다.

---

## 2. 패키지 및 의존성 설치 (Install the Package and Dependencies)

필요 패키지(Requirements)를 설치한 후, 로컬 `so101_gym` 패키지를 편집 가능(Editable, 개발자) 모드로 설치합니다. 이 과정에서 필요한 모든 라이브러리가 자동으로 함께 설치됩니다:

`pip install -r hw2_robot_control_mdps/requirements.txt` \
`pip install -e hw2_robot_control_mdps`

### 설치되는 패키지 목록

| 패키지 (Package) | 용도 (Purpose) |
|---------------------|-------------------------------------------------|
| `mujoco`            | 물리 시뮬레이션 엔진(Physics simulation engine) + 내장 뷰어 |
| `gymnasium`         | 강화 학습(RL) 환경 API (`SO100TrackEnv`에서 사용) |
| `stable-baselines3` | 훈련 및 평가를 위한 강화 학습 알고리즘 (PPO) |
| `tensorboard`       | 훈련 로그 시각화 |

---

## 3. 설치 확인 (Verify the Installation)

간단한 작동 테스트(Smoke test)를 진행합니다. 아래 명령어를 실행하면 SO-100 로봇 팔이 포함된 MuJoCo 뷰어가 열려야 합니다:

**Linux & Windows:**
`python scripts/interactive.py`

로봇이 표시되는 3D 뷰어 창이 열린다면 모든 설정이 정상적으로 완료된 것입니다. 다음 조작법을 통해 환경과 상호작용할 수 있습니다:
- **회전 (Rotate):** 클릭 후 드래그
- **줌 (Zoom):** 마우스 스크롤 휠
- **이동 (Pan):** Shift + 클릭 후 드래그

종료하려면 뷰어 창을 닫으세요.

---

## 4. 사용 가능한 스크립트 (Available Scripts)

| 스크립트 (Script) | 설명 (Description) | 명령어 (Command) |
|---------------------------|----------------------------------------------|----------------------------------|
| `scripts/interactive.py`  | 로봇을 확인하기 위해 MuJoCo 뷰어 실행 | `python scripts/interactive.py`  |
| `scripts/train.py`        | PPO 에이전트 훈련 (16개 병렬 환경 사용) | `python scripts/train.py`        |
| `scripts/evaluate_rand_targets.py`     | 뷰어에서 훈련된 정책(Policy) 평가 | `python scripts/evaluate_rand_targets.py`     |

---

## 5. 훈련 모니터링 (Monitor Training)

훈련 진행 중이거나 훈련이 끝난 후에 TensorBoard를 사용하여 지표를 시각화할 수 있습니다:

**Linux & Windows:**
`tensorboard --logdir logs/ --port 6006`

그 후 브라우저에서 http://localhost:6006 주소로 접속합니다.

---

## 문제 해결 (Troubleshooting)

| 문제 (Problem) | 해결 방법 (Fix) |
|---------|-----|
| `ModuleNotFoundError: No module named 'mujoco'` | 가상 환경(venv)이 활성화되어 있는지, 그리고 `pip install -r requirements.txt`를 실행했는지 확인하세요. |
| MuJoCo 뷰어가 열리지 않음 / EGL 오류 발생 *(Linux)* | Mesa/EGL 드라이버를 설치하세요: `sudo apt install libegl1-mesa-dev libgl1-mesa-dri libglvnd-dev` |
| `ERROR: could not create window` *(Linux)* | `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` 명령어를 실행하세요. |
| 뷰어 창이 검은색으로 나옴 *(Windows)* | 시스템의 GPU 드라이버를 업데이트하세요. |
| `ModuleNotFoundError: No module named 'env'` | `hw2_so100_tutorial/` 디렉터리에서 스크립트를 실행하거나, `PYTHONPATH=.`이 설정되어 있는지 확인하세요. 또한 `pip install -e .`을 실행했는지 확인하세요. |
| `python` 명령어를 찾을 수 없음 *(Windows)* | 대신 `python3`를 사용하거나, 시스템 환경 변수(PATH)에 Python을 추가하세요. |
| `pip install` 실패 *(Windows)* | 가상 환경이 활성화되어 있는지 확인하세요 (`.\mujoco\Scripts\Activate.ps1`). |