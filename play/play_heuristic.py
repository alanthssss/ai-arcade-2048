# play/play_heuristic.py

from envs.game_2048 import Game2048Env
from agents.heuristic_agent import HeuristicAgent


def main():
    env = Game2048Env()
    agent = HeuristicAgent()

    obs, info = env.reset()
    done = False
    step = 0

    while not done:
        env.render()
        action = agent.choose_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
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