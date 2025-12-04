# eval/evaluate_dqn_2048.py

import argparse
import os
import statistics
from collections import Counter

from stable_baselines3 import DQN

from envs.game_2048 import Game2048Env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval a trained DQN model on 2048."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="要评估的模型路径，例如 models/dqn_2048_exp1_steps200000_xxx.zip",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="评估局数，默认 50 局。",
    )
    return parser.parse_args()


def evaluate(model_path: str, n_episodes: int = 50):
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    env = Game2048Env()
    model = DQN.load(model_path)

    scores = []
    max_tiles = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        scores.append(info["score"])
        max_tiles.append(int(env.board.max()))

    # 基本统计
    avg_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    best_score = max(scores)
    worst_score = min(scores)

    counter_tiles = Counter(max_tiles)

    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print("-" * 60)
    print(f"Avg score   : {avg_score:.2f}")
    print(f"Median score: {median_score:.2f}")
    print(f"Best score  : {best_score}")
    print(f"Worst score : {worst_score}")
    print("-" * 60)
    print("Max tile distribution:")
    for tile, cnt in sorted(counter_tiles.items()):
        print(f"  {tile:5d}: {cnt:3d}  ({cnt / n_episodes * 100:5.1f}%)")
    print("=" * 60)


def main():
    args = parse_args()
    evaluate(args.model_path, args.episodes)


if __name__ == "__main__":
    main()