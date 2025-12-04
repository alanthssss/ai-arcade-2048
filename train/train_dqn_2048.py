# train/train_dqn_2048.py

import argparse
import os
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.game_2048 import Game2048Env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train / continue training DQN on 2048."
    )
    parser.add_argument(
        "--experiment-name",
        dest="experiment_name",
        type=str,
        default="dqn_2048_exp1",
        help="实验名，用来区分不同超参数/版本。",
    )
    parser.add_argument(
        "--total-timesteps",
        dest="total_timesteps",
        type=int,
        default=200_000,
        help="本次训练的步数（这一次要再跑多少步）。",
    )
    parser.add_argument(
        "--continue-from",
        dest="continue_from",
        type=str,
        default=None,
        help="要从哪个已有模型继续训练（models/*.zip），留空则从头训练。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # 日志目录：logs/<experiment-name>/
    tb_log_dir = os.path.join("logs", args.experiment_name)

    # --- 创建环境 ---
    env = Game2048Env()

    # --- 新训练 or 续训 ---
    if args.continue_from:
        # 从已有模型加载，继续训练
        print(f"Loading model from: {args.continue_from}")
        model = DQN.load(
            args.continue_from,
            env=env,
            tensorboard_log=tb_log_dir,
        )
        # 续训：不重置 time steps，方便 TensorBoard 曲线连续
        reset_num_timesteps = False
    else:
        # 从零开始训练
        print("Training new model from scratch.")
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=100_000,
            batch_size=256,
            gamma=0.99,
            target_update_interval=1_000,
            train_freq=4,
            learning_starts=1_000,
            verbose=1,
            tensorboard_log=tb_log_dir,
        )
        reset_num_timesteps = True

    # --- checkpoint 回调：中途自动存档 ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix=args.experiment_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # --- 开始训练 ---
    print(
        f"Start learning: total_timesteps={args.total_timesteps}, "
        f"experiment={args.experiment_name}, "
        f"continue_from={args.continue_from}"
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        log_interval=100,
        reset_num_timesteps=reset_num_timesteps,
        callback=[checkpoint_callback],
    )

    # 训练后 model.num_timesteps 是“整个生命周期”的步数
    total_steps = int(model.num_timesteps)
    ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    model_path = os.path.join(
        "models", f"{args.experiment_name}_steps{total_steps}_{ts_str}.zip"
    )
    model.save(model_path)
    print(f"Training finished, model saved to: {model_path}")


if __name__ == "__main__":
    main()