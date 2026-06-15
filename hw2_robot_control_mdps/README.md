# 과제 2: 로봇 제어와 MDP

이 과제는 자기주도 학습용으로 정리한 버전입니다. 각 실습의 TODO를 채우고, 제공된 실행 스크립트로 로봇 동작과 평가 지표를 직접 확인하세요.

# Installation

[Installation_Guide.md](Installation_Guide.md)의 지침에 따라 필요한 패키지 및 의존성을 설치하십시오.

이 설정은 다양한 Linux 배포판과 Windows에서 테스트되었으며, 설치 가이드에서 이를 다루고 있습니다. macOS에서 MuJoCo 렌더링 문제가 발생한다면 Linux, WSL, Docker 같은 대체 환경을 사용하는 것도 방법입니다.
실습은 CPU 환경에서 테스트되었으므로 GPU 접근은 필요하지 않습니다.

# 이론 질문 활용 방법

이론 질문은 주제에 대한 이해를 스스로 확인하기 위한 것입니다. 답변은 짧고 직접적으로 작성해도 충분합니다.

# 실습 1: 작업 공간의 키포인트 설계 및 역기구학(Inverse Kinematics) 구현

이 실습에서는 3D 작업 공간(Workspace)에서 베르누이 레미니스케이트(Lemniscate of Bernoulli, 무한대 기호) 형태의 키포인트(Keypoint) 세트를 설계하고, 로봇의 말단 장치(End-effector)가 이 키포인트들을 추적할 수 있도록 로봇의 관절 공간(Joint space)을 얻기 위한 역운동학(Inverse Kinematics, IK)을 구현합니다.

## TODOs
### Code Implementation
`exercises/ex1.py` 파일의 TODO를 채우십시오:
1. **2D 레미니스케이트 키포인트 생성:** `get_lemniscate_keypoint(t, a)` 함수를 구현합니다. 이 함수는 타임 스텝 t에 따라 2D 레미니스케이트 곡선 위의 단일 키포인트 또는 키포인트 배열을 반환합니다.
2. **3D 키포인트 세트 생성:** 이전에 구현한 `get_lemniscate_keypoint(t, a)`를 사용하여 `build_keypoints(count, width, x_offset, z_offset)`에서 레미니스케이트 곡선 위의 완전한 3D 키포인트 세트를 구축합니다.
3. **역운동학 구현:** 감쇠 최소자승법(Damped Least Squares)을 사용하여 역운동학을 구현합니다. `ik_track` 함수의 모든 `TODO`를 완료하십시오.

완료한 후, 다음 명령어를 실행하여 역운동학을 사용해 로봇이 키포인트를 추적하는 결과를 테스트할 수 있습니다:

```bash
python scripts/inverse_kinematics.py
```

이 추적은 단순히 관절 위치를 IK 출력값으로 텔레포트(Teleport)하여 수행되며, 아직 제어(Control)는 관여하지 않습니다.

### Theoretical questions
1. 레미니스케이트의 너비를 늘리면(a를 증가시킴), 로봇이 IK를 수행할 때 어떤 문제가 발생할 수 있습니까?
2. IK에서 dt 매개변수를 변경하면 어떤 일이 발생할 수 있습니까?
3. 우리는 간단한 수치적(Numerical) IK 솔버를 구현했습니다. 해석적(Analytical) IK 솔버와 비교했을 때 장단점은 무엇입니까?
4. 최첨단(State-of-the-art) IK 솔버들과 비교했을 때 우리 IK 솔버의 한계는 무엇입니까?

이론 질문은 짧고 직접적인 답변만 요구합니다. 각 질문은 1문장으로 답변하는 것을 권장합니다.

## 기성(off-the-shelf) IK 솔버 예시
https://github.com/stack-of-tasks/pinocchio

https://github.com/kevinzakka/mink

## 자기 점검
1. `exercises/ex1.py`의 TODO를 채운 뒤 `python scripts/inverse_kinematics.py`가 정상 실행되는지 확인합니다.
2. 로봇이 생성된 키포인트를 따라가는지 관찰합니다.
3. 이론 질문에 짧게 답해 봅니다.


# 실습 2: 궤적 생성과 PID 제어

이 실습에서는 중간 로봇 움직임을 제어하면서 경유지(Waypoint) 궤적을 생성하는 방법을 배웁니다. 먼저, 부드러운 움직임을 보장하는 경유지를 생성하는 방법을 배우고, 이어서 궤적을 따르기 위한 제어 법칙(Control law)을 정의합니다.

### The Control Pipeline
여러분의 과제는 경유지 생성, 역운동학(IK) 계산, PID 제어(PID Control)로 구성된 파이프라인의 일부를 채우는 것입니다. 파이프라인의 구조는 다음과 같습니다:

1. **경유지 생성(Waypoint Generation):** 대상 키포인트 세트를 기반으로, 강의 중에 소개된 5차 스플라인(Quintic spline) 시간 스케일링을 사용하여 중간 경유지 궤적을 생성합니다.
2. **역운동학(Inverse Kinematics, IK):** 이전 실습에서 구현한 것으로, 작업 공간의 경유지를 관절 공간으로 매핑합니다.
3. **PID 제어(PID Control):** 원하는 관절 각도와 현재 관절 각도의 차이(추적 오차)가 PID 제어기에 입력됩니다. 제어기는 적절한 제어 신호를 계산합니다.
4. **MuJoCo 시뮬레이션:** 생성된 제어 신호는 `data.ctrl[:]`를 통해 관절 토크로서 모터에 직접 적용됩니다. 물리 엔진이 한 단계 진행되어 대상 경유지를 향한 로봇의 움직임을 시뮬레이션합니다.

전체 시스템은 다음과 같이 시각화할 수 있습니다:

```text
╭──────────────────────────────────────────────────╮
│            Trajectory & IK Generation            │
│  ╭────────────────────────────────────────────╮  │
│  │ 1. Generate Quintic Spline Waypoints       │  │
│  │ 2. Solve Inverse Kinematics (target_qpos)  │  │
│  ╰─────────────────────┬──────────────────────╯  │
╰────────────────────────┼─────────────────────────╯
                         │
      Target Joint Pos   │  Current Joint Pos
                         ▼
╭────────────────────────┴─────────────────────────╮
│               PID Controller                     │
│                                                  │
│  ╭──────────────────╮     ╭───────────────────╮  │
│  │ Compute Error    │     │ Calculate signal  │  │
│  │(target - current)│     │(Kp*P + Ki*I + Kd*D)│ │
│  ╰─────────▲────────╯     ╰─────────┬─────────╯  │
│            │                        │            │
│            │   ╭────────────────╮   │            │
│            ╰───┤ Apply Control  │◄──╯            │
│                │ (data.ctrl[:]) │                │
│                ╰────────┬───────╯                │
│                         ▼                        │
│                ╭────────────────╮                │
│                │  MuJoCo Sim    │                │
│                │  (mj_step)     │                │
│                ╰────────────────╯                │
╰──────────────────────────────────────────────────╯
```

## TODOs
### Code Implementation
`exercises/ex2.py` 파일의 TODO를 채우십시오:
1. **경유지 생성:** `generate_quintic_spline_waypoints(start, end, num_points)` 함수를 구현합니다. 이 함수는 정규화된 시간 `s`(0에서 1까지)를 받아 다항식을 사용하여 부드럽게 처리된 스케일링 인자를 반환합니다. 이 실습에 필요한 다항식은 두 번째 강의 슬라이드에서 찾을 수 있습니다.
2. **PID 제어:** `pid_control(...)`에서 비례(P), 적분(I), 미분(D) 항의 계산을 구현합니다. 결과 제어 신호는 두 번째 강의 슬라이드에서 찾을 수 있는 PID 공식에 따라 계산됩니다.

기본 게인(Gain) 값은 이미 다음과 같이 조정해 두었습니다:

```python
KP = 150.0, KI = 0.0, KD = 0.01
```
PID 제어기를 구현하기 전이라도 5차 스플라인 경유지 생성을 테스트할 수 있습니다. 5차 스플라인 경유지 생성을 테스트하려면 다음을 실행하십시오:

```bash
python scripts/quintic_spline.py
```
아직 PID 제어기를 정의하지 않았기 때문에 로봇이 실제로 제어 법칙에 따라 움직이지 않고 단순히 새로운 위치로 텔레포트되는 것을 볼 수 있습니다.

PID 제어기를 구현한 후에는 다음을 실행하여 전체 시스템을 테스트할 수 있습니다:

```bash
python scripts/pid_control.py
```

로봇이 미리 정의된 여러 키포인트 사이를 부드럽게 이동하는 모습을 보여주는 뷰어 창이 나타나야 합니다.

### Theoretical questions
PID 게인 선택에 대한 감을 잡기 위해, 게인 선택이 경유지 추적 동작에 어떤 영향을 미치는지 분석합니다.
다음 질문에 답할 수 있도록 게인의 다양한 설정을 테스트해 보십시오:
1. $K_P$를 계속 증가시키면 경유지를 추적할 때 어떤 문제가 발생합니까?
2. $K_D$는 위에서 $K_P$를 증가시켰을 때 나타난 효과를 어떻게 완화합니까?
3. 제어기가 잘 작동하기 위해 0이 아닌 $K_I$가 필요한 시나리오는 무엇입니까?

이론 질문은 짧고 직접적인 답변만 요구합니다. 각 질문은 1문장으로 답변하는 것을 권장합니다.

## 자기 점검
1. `exercises/ex2.py`의 TODO를 채운 뒤 `python scripts/pid_control.py`가 정상 실행되는지 확인합니다.
2. 로봇이 경유지 사이를 부드럽게 이동하는지 관찰합니다.
3. PID 게인을 바꿔 보며 이론 질문에 답해 봅니다.


# 실습 3: 경유점(Waypoint) 추종 정책 학습

이 실습에서는 마르코프 결정 과정(Markov Decision Process, MDP)의 시뮬레이션을 다루고, 무작위로 선택된 경유지를 추적하도록 강화 학습(Reinforcement Learning, RL) 정책(Policy)을 훈련합니다.
이전 실습과 비교하여 파이프라인도 변경되었으며 다음과 같은 구조를 가집니다:

1. **관측(Observation):** 환경은 현재 환경 상태(로봇 및 목표)를 포함하는 관측 벡터를 구성합니다. 이는 RL 에이전트(Agent)에게 전송됩니다.
2. **행동 예측(Action Prediction):** RL 에이전트는 **10 Hz**(0.1초마다)의 제어 주기로 `[-1, 1]` 범위의 정규화된 행동을 예측합니다.
3. **행동 처리(Action Processing):** 환경은 정규화된 행동을 수신하여 로봇의 물리적 한계 내에서 관절 위치 목표로 변환합니다.
4. **Mujoco 시뮬레이션 단계:** 이러한 목표 위치는 `data.ctrl`에 입력되며 MuJoCo 내부적으로 목표 관절 토크로 변환됩니다. Mujoco 시뮬레이터는 0.002초(500 Hz)마다 단계를 진행하며, 환경이 RL 제어기로부터 새로운 행동을 요구하기 전에 50번의 단계(`ctrl_decimation = 50`)를 진행합니다(10 Hz).
5. **보상 계산(Reward Calculation):** 각 제어 단계(또는 50번의 시뮬레이션 단계) 후에 새로운 말단 장치 추적 오차가 측정되고, 보상이 계산되며, 주기를 반복하기 위해 새로운 상태가 RL 에이전트로 반환됩니다.

전체 시스템은 다음과 같이 시각화할 수 있습니다:

```text
╭──────────────────────────────────────────────────╮
│                  PPO Agent                       │
│  ╭────────────────────────────────────────────╮  │
│  │             Policy Network                 │  │
│  ╰─────────────────────┬──────────────────────╯  │
╰────────────────────────┼─────────────────────────╯
                         │
        Observation      │  Action [-1, 1]
   (Joints, EE, Target)  │  (10 Hz)
                         ▼
╭────────────────────────┴─────────────────────────╮
│               RL Environment                     │
│                                                  │
│  ╭──────────────────╮     ╭───────────────────╮  │
│  │ Get Observation  │     │ Process Action    │  │
│  │ (State Vector)   │     │ (Scale to Limits) │  │
│  ╰─────────▲────────╯     ╰─────────┬─────────╯  │
│            │                        │            │
│  ╭─────────┴────────╮     ╭─────────▼─────────╮  │
│  │ Calculate Reward │     │ Set Action Targets│  │
│  │ (Dist to Target) │     │ (data.ctrl[:] = a)│  │
│  ╰─────────▲────────╯     ╰─────────┬─────────╯  │
│            │                        │            │
│            │   ╭────────────────╮   │            │
│            ╰───┤  Control Loop  |◄──╯            |
|                |    (10 Hz)     │                │
│                ╰────────┬───────╯                │
│                         ▼                        │
│                ╭────────────────╮                │
│                │     MuJoCo     │                │
│                │    (500 Hz)    │                │
│                ╰────────────────╯                │
╰──────────────────────────────────────────────────╯
```
## TODOs
이제 `exercises/ex3.py` 파일의 TODO를 채우십시오:
1. **로봇 초기화(Reset Robot):** 약간의 무작위 노이즈를 주어 로봇을 기본 관절 위치로 초기화하는 함수를 구현합니다.
2. **목표 위치 초기화(Reset Target Position):** 로봇의 베이스(Base)를 기준으로 무작위 3D 목표 위치를 샘플링하고 계산합니다.
3. **행동 처리:** 정규화된 정책 행동(`[-1, 1]` 범위)을 주어진 범위 내의 목표 관절 위치로 변환합니다.
4. **보상 계산:** 말단 장치와 목표 사이의 거리(오차)를 기반으로 RL 보상을 계산합니다.
5. **관측값 획득(Get Observation):** 환경 상태에서 관측 벡터를 추출합니다. 일부 상태를 월드 프레임(World frame)에서 베이스 프레임(Base frame)으로 변환하기 위해 좌표 변환을 수행해야 함에 유의하십시오.

이 과제를 완료하는 데 필수적인 것은 아니지만, 로봇 공학에서 사용되는 3D 공간에서의 회전 표현과 MuJoCo의 관절 위치 `qpos` 및 기타 매개변수의 차원 수에 대한 논리가 궁금하다면 https://eater.net/quaternions 를 확인해 보십시오.

이제 환경 블록을 구현했으므로, `scripts/train.py`를 사용하여 정책을 훈련해 보겠습니다. 훈련 인자(Arguments)는 `scripts/train.py`를 참조하십시오. `--max_iterations` 및 `--save_checkpt_freq` 외에는 인자를 수정할 필요가 없습니다. 스크립트가 실행되는 동안 새 터미널을 열고 hw2_so100_tutorial 폴더에서 다음을 실행하여 훈련 진행 상황과 보상을 실시간으로 추적할 수 있습니다:

`tensorboard --logdir=logs --port=6006`

그 후 브라우저에서 제공된 링크(http://localhost:6006)로 텐서보드(TensorBoard)를 여십시오.

체크포인트는 자동으로 저장됩니다. 평가를 위해 `scripts/evaluate_rand_targets.py`에서 특정 체크포인트를 선택하십시오. 예를 들어, 다음을 실행합니다:

```bash
python scripts/evaluate_rand_targets.py --load_run=1 --checkpoint=500
```

시뮬레이션 창이 나타납니다. 로봇은 10개의 에피소드(각 에피소드는 2초로 설정됨) 동안 추적을 수행합니다. 최종 ee_tracking_error가 터미널에 출력되며, 10개 에피소드가 끝날 때 평균값이 포함됩니다.
자기 점검 기준으로 **평균 최종 EE 추적 오차 < 0.05**를 목표로 삼아 보세요. 정책이 제대로 수렴한다면 이는 쉽게 달성할 수 있습니다.

### Theoretical questions (Bonus)
훈련된 정책을 배포했을 때 로봇의 성능을 관찰하십시오. 다음 코드를 실행하여 RL 정책의 성능을 이전에 구현한 PID 제어기와 비교할 수 있습니다:

```bash
python scripts/evaluate_trajectory.py --load_run=1 --checkpoint=500
```

로봇이 레미니스케이트 곡선 위의 키포인트를 추적할 때 어떤 차이점을 관찰할 수 있습니까? RL 정책의 성능을 향상시키기 위해 ex3의 함수들을 어떻게 변경할 수 있습니까? 이 함수들을 수정하십시오(함수의 인자를 변경하고 `env/so100_tracking_env.py`에서도 그에 맞게 변경할 수 있습니다). 새로운 환경으로 다른 RL 정책을 훈련하고 성능 변화를 비교하며, 여러분의 변경 사항이 로봇의 성능에 어떤 영향을 미쳤는지 설명하십시오. PPO 하이퍼파라미터(gamma, ent_coef 등)를 변경할 수도 있습니다.

## 자기 점검
1. `exercises/ex3.py`의 TODO를 채운 뒤 정책 학습과 평가 스크립트가 정상 실행되는지 확인합니다.
2. 터미널에 출력되는 평균 추적 오차를 기록하고, 목표 기준에 가까워지는지 확인합니다.
3. 추가 실험을 했다면 변경한 환경/보상/하이퍼파라미터와 성능 변화를 짧게 정리합니다.

### Important Concepts
- `np.clip(a, a_min, a_max)`: 배열의 값을 주어진 범위 내로 제한합니다(관절 위치를 제한하는 데 유용함).
- 제자리 배열 수정(In-place Array Modification): 데이터 객체(예: `mujoco.MjData`)의 속성인 배열을 수정할 때, 물리 엔진이 변경 사항을 감지할 수 있도록 전체 배열 참조를 덮어쓰는 대신 슬라이싱(예: `[:]`)을 사용하여 배열을 제자리에서 수정하십시오.


## Further Resources

이 실습에서는 전체 환경 클래스나 강화 학습 알고리즘의 내부 작동 방식을 노출하지 않습니다. 향후 강의에서 다른 연속적 행동 공간(Continuous action space) 및 정책 그래디언트(Policy gradient) 방법과 함께 적용된 RL 알고리즘인 PPO(Proximal Policy Optimization)에 익숙해질 것입니다. 그러나 향후 실습을 준비하기 위해 Mujoco, gymnasium 및 PPO 알고리즘(stable-baselines3 제공)의 작동 방식을 이해할 수 있도록 전체 코드베이스를 살펴보는 것을 강력히 권장합니다.

이 설정에 대해 깊이 있게 탐구하고 싶다면 다음 리소스를 참고하십시오:

* **PPO의 이론적 배경:**
  * [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
* **사용된 PPO 구현체 문서:**
  * [Stable-Baselines3: PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
* **물리 엔진 문서:**
  * [MuJoCo Documentation](https://mujoco.readthedocs.io/)
* **고급 환경:**
  * 더 고급스럽고 확장 가능한 환경으로 NVIDIA Isaac Lab이 있으며, ETH 실험실에서 프로젝트를 수행하는 분들에게 친숙할 것입니다: [NVIDIA Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)

다중 프로세싱(Multiprocessing)의 이점을 얻기 위해 여러 환경을 병렬로 생성하고 있다는 사실 등 추가적인 기술적 세부 사항은 공개되지 않았습니다. 다행히도 이 실습의 목적상 구현된 함수들은 별도의 스레드에서 작동하므로 이러한 병렬성을 고려할 필요가 없으며, 함수들이 단일 환경에서 작동한다고 가정해도 무방합니다.
