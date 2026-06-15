PPO_PARAMETERS = {
    "seed": 42,
    "hidden_sizes": [256, 128, 128],
    "total_iterations": 500, # 총 학습 반복(iteration) 횟수
    "n_steps": 2048, # 매 업데이트 전 환경 단계(env steps) 수
    "mini_batch_size": 1024, # PPO 업데이트를 위한 배치 크기(batch size)
    "n_epochs": 10, # PPO 업데이트당 에포크(epoch) 수
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "surrogate_loss_coeff": 1.0,
    "value_loss_coeff": 0.01,
    "entropy_coeff": 0.005,
    "clip_ratio": 0.2,
    "learning_rate": 3e-4,
    "target_kl": 0.01,
    "max_grad_norm": 0.5,
    "save_interval": 50,
}