import argparse

import gymnasium as gym

from .common import ALGOS, MODELS, RUNS, load_cfg


def train(cfg: dict):
    env = gym.make(cfg["env"])
    Algo = ALGOS[cfg["algo"]]
    model = Algo(cfg["policy"], env, seed=cfg["seed"], verbose=1,
                 tensorboard_log=str(RUNS), **cfg.get("hyperparams", {}))
    model.learn(total_timesteps=cfg["total_timesteps"], tb_log_name=cfg["algo"])
    MODELS.mkdir(exist_ok=True)
    model.save(MODELS / cfg["algo"])
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    train(load_cfg(ap.parse_args().config))
