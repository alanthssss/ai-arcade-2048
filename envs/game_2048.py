# envs/game_2048.py

import random
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Game2048Env(gym.Env):
    """
    2048 游戏的 Gymnasium 环境。

    状态：
        4x4 棋盘，每个格子是 log2(tile_value)，空格为 0。
        比如：2 -> 1, 4 -> 2, 8 -> 3, ..., 2048 -> 11

    动作：
        0: 上, 1: 下, 2: 左, 3: 右
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0

        # 动作空间：上、下、左、右
        self.action_space = spaces.Discrete(4)

        # 观察空间：4x4，每个格子 [0, 16]（log2 后不会太大）
        self.observation_space = spaces.Box(
            low=0.0,
            high=16.0,
            shape=(4, 4),
            dtype=np.float32,
        )

    # ---------------- Gymnasium API ----------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.board[:] = 0
        self.score = 0
        self._add_random_tile()
        self._add_random_tile()
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        返回：(obs, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), "invalid action"

        old_board = self.board.copy()
        old_score = self.score

        gained = self._move(action)
        moved = not np.array_equal(old_board, self.board)

        if moved:
            self._add_random_tile()

        self.score += gained
        terminated = not self._can_move()  # 没法再走就结束
        truncated = False  # 不做时间截断

        reward = float(gained)
        obs = self._get_obs()
        info = {
            "score": self.score,
            "gained": gained,
        }
        return obs, reward, terminated, truncated, info

    # ---------------- 内部逻辑 ----------------

    def _get_obs(self) -> np.ndarray:
        """
        把棋盘转成 log2 形式，空格为 0。
        """
        obs = self.board.astype(np.float32)
        non_zero = obs > 0
        obs[non_zero] = np.log2(obs[non_zero])
        return obs

    def _add_random_tile(self) -> None:
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            return
        x, y = random.choice(empty)
        self.board[x, y] = 2 if random.random() < 0.9 else 4

    def _move(self, action: int) -> int:
        """
        执行一次动作（直接修改 self.board），并返回本次合成获得的分数。
        0: up, 1: down, 2: left, 3: right
        """
        rotated = np.rot90(self.board, k=action)
        score_gain = 0

        for col in range(4):
            line = rotated[:, col]
            new_line, gained = self._merge_line(line)
            rotated[:, col] = new_line
            score_gain += gained

        self.board = np.rot90(rotated, k=-action)
        return score_gain

    @staticmethod
    def _merge_line(line: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        单列合并逻辑：
        - 移除 0
        - 相邻相同合并
        - 填充 0 凑满 4 个
        返回：新的列 & 本列获得的分数
        """
        line = np.array(line, dtype=np.int32)
        non_zero = line[line != 0]
        if len(non_zero) == 0:
            return np.zeros_like(line), 0

        merged = []
        score_gain = 0
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                v = non_zero[i] * 2
                merged.append(v)
                score_gain += v
                skip = True
            else:
                merged.append(non_zero[i])

        while len(merged) < 4:
            merged.append(0)

        return np.array(merged, dtype=np.int32), score_gain

    def _can_move(self) -> bool:
        if np.any(self.board == 0):
            return True
        # 没有空格了，看是否有相邻相等的
        for x in range(4):
            for y in range(4):
                v = self.board[x, y]
                if x + 1 < 4 and self.board[x + 1, y] == v:
                    return True
                if y + 1 < 4 and self.board[x, y + 1] == v:
                    return True
        return False

    def render(self) -> None:
        print("-" * 25)
        print(f"Score: {self.score}")
        for row in self.board:
            row_str = " ".join(f"{v:4d}" for v in row)
            print(row_str)
        print("-" * 25)