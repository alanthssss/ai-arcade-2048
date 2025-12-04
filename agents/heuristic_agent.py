# agents/heuristic_agent.py

import math
import numpy as np

from envs import Game2048Env


class HeuristicAgent:
    """
    非学习版启发式 AI：
    - 对每个动作进行模拟
    - 用启发式规则给结果棋盘打分
    - 选择评分最高的动作

    启发式评分包含：
    - 空格数量越多越好
    - 最大 tile 越大越好（log2）
    - 棋盘“单调性”越好越高（大概保证大数集中在一侧）
    """

    def choose_action(self, env: Game2048Env) -> int:
        best_score = -1e15
        best_action = 0

        for action in range(env.action_space.n):
            backup_board = env.board.copy()
            backup_score = env.score

            gained = env._move(action)
            moved = not np.array_equal(backup_board, env.board)

            if not moved:
                # 恢复现场
                env.board = backup_board
                env.score = backup_score
                continue

            # 模拟结果棋盘
            sim_board = env.board.copy()
            score = self._evaluate(sim_board) + gained * 1.0

            # 恢复现场
            env.board = backup_board
            env.score = backup_score

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _evaluate(self, board: np.ndarray) -> float:
        empty_count = np.sum(board == 0)
        max_tile = np.max(board) if board.size > 0 else 0

        score = 0.0
        score += empty_count * 10.0
        if max_tile > 0:
            score += math.log2(max_tile) * 50.0

        score += self._monotonicity(board) * 1.0

        return score

    def _monotonicity(self, board: np.ndarray) -> float:
        """
        简易单调性估计：行和列方向尽量从一端向另一端递减。
        """
        b = board.astype(np.float32)
        mask = b > 0
        b[mask] = np.log2(b[mask])

        mono_score = 0.0

        # 行方向
        for row in b:
            for i in range(3):
                if row[i] >= row[i + 1]:
                    mono_score += 1.0

        # 列方向
        for col in b.T:
            for i in range(3):
                if col[i] >= col[i + 1]:
                    mono_score += 1.0

        return mono_score