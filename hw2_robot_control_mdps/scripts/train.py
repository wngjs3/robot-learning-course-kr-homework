import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

from __init__ import *
from utils import *
from env.so100_tracking_env import SO100TrackEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on SO100 tracking")
    parser.add_argument("--num_envs", type=int, default=16,
                        help="Number of parallel environments; set 1 for single process")
    parser.add_argument("--max_iterations", type=int, default=500,
                        help="Number of PPO update iterations")
    parser.add_argument("--save_checkpt_freq", type=int, default=50,
                        help="Checkpoint every N update iterations")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu or cuda)")
    return parser.parse_args()

def make_env():
    def _init():
        env = SO100TrackEnv(xml_path=XML_PATH, render_mode=None)
        env = Monitor(env, info_keywords=("ee_tracking_error",))
        return env
    return _init

if __name__ == "__main__":
    args = parse_args()

    if args.num_envs > 1:
        # 병렬 시뮬레이션을 위해 벡터화된 환경(vectorized environment)으로 래핑
        start_method = "spawn" if sys.platform == "win32" else "forkserver"
        envs = SubprocVecEnv([make_env() for _ in range(args.num_envs)], start_method=start_method)
        envs = VecMonitor(envs)
        print(f"Successfully launched {args.num_envs} environments!")
        model = PPO(
            "MlpPolicy",
            envs,
            device=args.device,
            verbose=1,
            tensorboard_log=EXP_DIR,
            gamma=0.99,
            ent_coef=0.001,
            vf_coef=1.0
        )
    else:
        # 렌더링을 포함한 디버그용 단일 환경 생성
        env = SO100TrackEnv(xml_path=XML_PATH, render_mode="human")
        model = PPO(
            "MlpPolicy",
            env,
            device=args.device,
            verbose=1,
            tensorboard_log=EXP_DIR,
            gamma=0.99,
            ent_coef=0.001,
            vf_coef=1.0
        )

    update_checkpoint_callback = UpdateCheckpointCallback(
        save_path=None,
        name_prefix="model",
        save_freq_updates=args.save_checkpt_freq,
        verbose=0,
    )

    # 목표 PPO 업데이트 단계 수만큼 학습 (각 업데이트 = n_steps * n_envs 트랜지션)
    total_update_steps = args.max_iterations
    rollout_batch_size = model.n_steps * model.n_envs
    total_timesteps = total_update_steps * rollout_batch_size
    print(f"Training for {total_update_steps} iterations (~{total_timesteps} steps, rollout size {rollout_batch_size})")

    envs_ref = model.get_env()
    try:
        model.learn(total_timesteps=total_timesteps,
                    log_interval=1,
                    tb_log_name=EXP_NAME,
                    callback=[EpisodeLoggingCallback(),
                              update_checkpoint_callback,
                              KLAdaptiveLRCallback(target_kl=0.05, init_lr=1e-3, min_lr=1e-5, max_lr=1e-3)]
                    )
    finally:
        # 환경이 GL 컨텍스트를 깔끔하게 닫도록 보장
        if envs_ref is not None:
            envs_ref.close()

    # 최종 모델 저장
    log_dir = model.logger.get_dir()
    if log_dir is None:
        raise ValueError("Logger directory is not set; cannot save final model")
    final_update_step = update_checkpoint_callback.update_counter
    final_model_path = Path(log_dir) / f"model_{final_update_step}"
    model.save(str(final_model_path))
    print(f"Saved models to {log_dir}")