"""
실습 2 (DQN)를 위한 하이퍼파라미터.

다음 항목들을 튜닝하는 것을 권장합니다:
- lr
- epsilon
- target_update
- hidden_dim

명시적으로 언급되지 않은 한 나머지 파라미터는 변경하지 않고 그대로 유지해 주세요.
"""

DQN_PARAMETERS = {
    # TODO: 다음 하이퍼파라미터를 튜닝하세요
    # 기본값을 원하는 값으로 변경하세요.
    "lr": 1e-3,            # TODO
    "epsilon": 0.03,       # TODO
    "target_update": 10,   # TODO
    "hidden_dim": 128,     # TODO
    
    # 고정된 파라미터
    "gamma": 0.99,
    "num_episodes": 500,
    "buffer_size": 10000,
    "minimal_size": 500,
    "batch_size": 64,
    "seed": 0,
}