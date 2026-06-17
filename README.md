# sandbox-gymnax

Small `gymnax` reinforcement-learning sandbox using JAX.

## Setup

```bash
uv sync
```

For development tools:

```bash
uv sync --group dev
```

## Train

List available environments:

```bash
uv run train
```

Train CartPole:

```bash
uv run train --env-id CartPole-v1
```

By default, `train` runs until the environment is solved: `mean_return >= threshold` over 100 eval episodes. Built-in thresholds are available for `CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`, and `Breakout-MinAtar`. For other environments, pass `--target-return`.

Train Acrobot:

```bash
uv run train --env-id Acrobot-v1 --output-dir outputs/acrobot
```

Quick capped smoke run:

```bash
uv run train --env-id CartPole-v1 --updates 2 --num-envs 8 --rollout-steps 16 --output-dir outputs/smoke
```

`--updates` is a max-update cap. Omit it for normal solve-until-done training.

The trainer writes:

- `checkpoint.pkl`: best evaluated policy parameters
- `checkpoint.json`: metadata used by `play`, including `env_id`
- `metrics.json`: training and eval history
- `params.pkl`: legacy copy of the checkpoint parameters

## Play

`play` only needs the checkpoint. It reads the sibling metadata file to determine which environment to run.

Open a Python GUI window:

```bash
uv run play outputs/cartpole/checkpoint.pkl
uv run play outputs/acrobot/checkpoint.pkl --seed 40
uv run play outputs/mountaincar/checkpoint.pkl
```

Controls: space pauses/resumes, arrow keys step backward/forward, `r` resets, and `q` or escape quits.

Write a GIF:

```bash
uv run play outputs/acrobot/checkpoint.pkl --seed 40 --mode gif --output outputs/acrobot/rollout.gif
```

Write a browser player:

```bash
uv run play outputs/acrobot/checkpoint.pkl --seed 40 --mode browser --open
```

Browser playback is written to `outputs/<env>/playback/player.html`.

## Breakout-MinAtar Solver

```bash
uv run solve-breakout-minatar
```

The solver evaluates an observation-based Breakout-MinAtar policy over 256 episodes, writes metrics to `outputs/breakout_minatar/solver_metrics.json`, and renders a GIF to `outputs/breakout_minatar/breakout_minatar.gif`. The default target return is `30`, corresponding to clearing the initial 30-brick board at least once on average.

## Legacy Helpers

These wrappers still work:

```bash
uv run train-cartpole
uv run play-cartpole
uv run play-acrobot
uv run play-gui --game cartpole --open
```
