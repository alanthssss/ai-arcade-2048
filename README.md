# AI Arcade - 2048

一个用于实验「AI 玩 2048」的最小项目：

- `envs/game_2048.py`：2048 游戏环境（Gym 风格）
- `agents/heuristic_agent.py`：不用训练就能玩的启发式 AI
- `train/train_dqn_2048.py`：用 Stable-Baselines3 训练 DQN
- `play/play_heuristic.py`：用启发式 AI 玩 2048（终端可视化）
- `play/play_dqn.py`：用训练好的 DQN 模型玩 2048

## 使用方法

```bash
# 安装依赖
pip install -r requirements.txt

# 启发式 AI 玩 2048
python -m play.play_heuristic

# 训练 DQN（会比较久，可以先调小 total_timesteps）
python -m train.train_dqn_2048

# 用 DQN 模型玩 2048
python -m play.play_dqn
# ai-arcade-2048
