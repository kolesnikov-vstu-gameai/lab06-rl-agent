import argparse
import json

import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from .common import ALGOS, MODELS, RESULTS


def evaluate(algo: str, env_id: str = "CartPole-v1", n: int = 20) -> dict:
    model = ALGOS[algo].load(MODELS / algo)
    mean, std = evaluate_policy(model, gym.make(env_id), n_eval_episodes=n)
    RESULTS.mkdir(exist_ok=True)
    out = {"algo": algo, "env": env_id, "episodes": n, "mean_reward": mean, "std_reward": std}
    (RESULTS / f"eval_{algo}.json").write_text(json.dumps(out, indent=2))
    print(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--env", default="CartPole-v1")
    a = ap.parse_args()
    evaluate(a.algo, a.env)
