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

# 多次训练
- 初次
python -m train.train_dqn_2048 \
  --experiment-name dqn_2048_v1 \
  --total-timesteps 200000
python -m eval.evaluate_dqn_2048 \
  --model-path models/dqn_2048_v1_steps200000_20251204-111341.zip \
  --episodes 50
- 后续
python -m train.train_dqn_2048 \
  --experiment-name dqn_2048_v1 \
  --total-timesteps 200000 \
  --continue-from models/dqn_2048_v1_steps200000_20251204-111341.zip
python -m eval.evaluate_dqn_2048 \
  --model-path models/dqn_2048_v1_steps400000_20251204-111902.zip \
  --episodes 50
python -m train.train_dqn_2048 \
  --experiment-name dqn_2048_v1 \
  --total-timesteps 200000 \
  --continue-from models/dqn_2048_v1_steps400000_20251204-111902.zip
python -m eval.evaluate_dqn_2048 \
  --model-path models/dqn_2048_v1_steps600000_20251204-112859.zip \
  --episodes 50
python -m train.train_dqn_2048 \
  --experiment-name dqn_2048_v1 \
  --total-timesteps 400000 \
  --continue-from models/dqn_2048_v1_steps600000_20251204-112859.zip
python -m eval.evaluate_dqn_2048 \
  --model-path models/dqn_2048_v1_steps1000000_20251204-113615.zip \
  --episodes 50
python -m train.train_dqn_2048 \
  --experiment-name dqn_2048_v1 \
  --total-timesteps 1000000 \
  --continue-from models/dqn_2048_v1_steps1000000_20251204-113615.zip
python -m eval.evaluate_dqn_2048 \
  --model-path models/dqn_2048_v1_steps2000000_20251204-114402.zip \
  --episodes 50
- 持续训练曲线
tensorboard --logdir logs

# DQN_X(dev)
- 初次
python -m train.train_dqn_2048 \
  --experiment-name dqn_2048_v1 \
  --total-timesteps 200000 \
  --eval-episodes 50 \
  --callback-eval-freq 50000 \
  --callback-eval-episodes 20
python -m train.train_dqn_2048 \
  --experiment-name dqn_2048_v1 \
  --total-timesteps 200000 \
  --eval-episodes 50 \
  --callback-eval-freq 50000 \
  --callback-eval-episodes 20
- 后续
python -m train.train_dqn_2048 \
  --experiment-name dqn_2048_v1 \
  --total-timesteps 200000 \
  --continue-from models/dqn_2048_v1_latest.zip \
  --eval-episodes 50 \
  --callback-eval-freq 50000 \
  --callback-eval-episodes 20
python -m train.train_dqn_2048 \
  --experiment-name dqn_2048_v1 \
  --total-timesteps 200000 \
  --continue-from models/dqn_2048_v1_latest.zip \
  --eval-episodes 50 \
  --callback-eval-freq 50000 \
  --callback-eval-episodes 20

- csv
python -m eval.evaluate_dqn_2048 \
  --model-path models/dqn_2048_v1_latest.zip \
  --episodes 50 \
  --csv-path eval_results.csv \
  --tag v1_2000k

