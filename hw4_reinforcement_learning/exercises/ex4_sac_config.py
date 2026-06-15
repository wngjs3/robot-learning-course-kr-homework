SAC_PARAMETERS = {
    "seed": 42,
    "hidden_sizes": [256, 128, 128],
    "total_iterations": 2048, # 총 훈련 반복 횟수
    "learning_start_steps": 1000, # 훈련을 시작하기 전에 수집할 환경 단계(env steps) 수
    "train_freq": 500, # SAC 업데이트 사이의 환경 단계(env steps) 수
    "gradient_steps": 200, # SAC 업데이트당 그래디언트 업데이트 횟수
    "batch_size": 1024, # SAC 업데이트를 위한 배치 크기
    "eval_freq": 2048, # 평가 사이의 환경 단계(env steps) 수
    "replay_size": 1_000_000,
    "gamma": 0.99,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "init_alpha": 0.2,
    "target_entropy": None,
    "save_interval": 50,
}