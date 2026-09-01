"""Сравнительный график кривых обучения из логов TensorBoard."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

from .common import RESULTS, RUNS  # noqa: E402

plt.figure()
for run in sorted(RUNS.glob("*")):
    ea = EventAccumulator(str(run)).Reload()
    if "rollout/ep_rew_mean" not in ea.Tags()["scalars"]:
        continue
    ev = ea.Scalars("rollout/ep_rew_mean")
    plt.plot([e.step for e in ev], [e.value for e in ev], label=run.name)
plt.xlabel("timesteps")
plt.ylabel("mean episode reward")
plt.legend()
RESULTS.mkdir(exist_ok=True)
plt.savefig(RESULTS / "learning_curves.png", dpi=150)
print("→", RESULTS / "learning_curves.png")
