# play/play_dqn.py

from stable_baselines3 import DQN

from envs.game_2048 import Game2048Env


def main():
    env = Game2048Env()
    model = DQN.load("dqn_2048_final")

    obs, info = env.reset()
    done = False
    step = 0

    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        step += 1
        print(
            f"Step {step}, action={action}, "
            f"gained={reward}, total_score={info['score']}"
        )

    env.render()
    print("Game over!")
    print("Final score:", info["score"])
    print("Max tile:", env.board.max())


if __name__ == "__main__":
    main()