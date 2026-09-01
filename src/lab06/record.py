import argparse

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from .common import ALGOS, MODELS, RESULTS


def record(algo: str, env_id: str = "CartPole-v1", episodes: int = 5):
    env = RecordVideo(gym.make(env_id, render_mode="rgb_array"), str(RESULTS / "videos" / algo),
                      episode_trigger=lambda e: True)
    model = ALGOS[algo].load(MODELS / algo)
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    record(ap.parse_args().algo)
