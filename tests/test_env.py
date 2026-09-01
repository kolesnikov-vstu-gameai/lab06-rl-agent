import gymnasium as gym


def test_env_step():
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,)
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    assert r == 1.0
