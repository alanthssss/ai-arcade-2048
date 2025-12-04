# train/train_dqn_2048.py

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from envs import Game2048Env


def main():
    env = Game2048Env()

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
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="dqn_2048",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # 提示：第一次可以先把 total_timesteps 改小试试，例如 200_000
    model.learn(
        total_timesteps=200_000,
        log_interval=100,
        callback=[checkpoint_callback],
    )

    model.save("dqn_2048_final")
    print("Training finished, model saved as dqn_2048_final.zip")


if __name__ == "__main__":
    main()
