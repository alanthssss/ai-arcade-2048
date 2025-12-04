# eval/evaluate_dqn_2048.py

import argparse
import csv
import os
import statistics
from collections import Counter
from typing import Optional, Dict, Any

from stable_baselines3 import DQN

from envs.game_2048 import Game2048Env


def evaluate(
    model_path: str,
    n_episodes: int = 50,
    csv_path: Optional[str] = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    评估一个 DQN 模型在 2048 上的表现。

    :param model_path: 模型文件路径
    :param n_episodes: 评估局数
    :param csv_path: 可选，把结果追加到 CSV 文件
    :param tag: 可选，额外标记（比如 'v1_200k', 'v1_600k'）
    :return: 一个包含统计信息的 dict
    """
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

    avg_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    best_score = max(scores)
    worst_score = min(scores)

    counter_tiles = Counter(max_tiles)

    print("=" * 60)
    print(f"Model: {model_path}")
    if tag:
        print(f"Tag  : {tag}")
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

    result = {
        "model_path": model_path,
        "tag": tag or "",
        "episodes": n_episodes,
        "avg_score": avg_score,
        "median_score": median_score,
        "best_score": best_score,
        "worst_score": worst_score,
        "max_tile_counts": dict(counter_tiles),
    }

    # 如果指定了 csv_path，就把结果追加进去
    if csv_path:
        header = [
            "model_path",
            "tag",
            "episodes",
            "avg_score",
            "median_score",
            "best_score",
            "worst_score",
            "max_tile_counts",
        ]
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            row = {
                "model_path": model_path,
                "tag": tag or "",
                "episodes": n_episodes,
                "avg_score": avg_score,
                "median_score": median_score,
                "best_score": best_score,
                "worst_score": worst_score,
                "max_tile_counts": dict(counter_tiles),
            }
            writer.writerow(row)
        print(f"[Eval] Result appended to CSV: {csv_path}")

    return result


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Eval a trained DQN model on 2048."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="要评估的模型路径，例如 models/dqn_2048_v1_steps200000_xxx.zip",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="评估局数，默认 50 局。",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="可选，把结果追加到指定 CSV 文件。",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="可选，为这次评估加一个标记（例如 v1_200k）。",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        csv_path=args.csv_path,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
