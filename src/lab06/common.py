from pathlib import Path

import yaml
from stable_baselines3 import A2C, DQN, PPO, SAC

ROOT = Path(__file__).resolve().parents[2]
RUNS, MODELS, RESULTS = ROOT / "runs", ROOT / "models", ROOT / "results"
ALGOS = {"ppo": PPO, "dqn": DQN, "a2c": A2C, "sac": SAC}


def load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
