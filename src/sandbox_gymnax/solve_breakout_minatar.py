from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw


def expert_actions(obs: jax.Array) -> jax.Array:
    """Move the paddle toward the current ball x-position.

    Breakout-MinAtar's minimal action ids are 0=noop, 1=left, 2=right.
    """
    paddle_columns = obs[..., 9, :, 0]
    ball_columns = jnp.max(obs[..., :, :, 1], axis=-2)
    paddle_x = jnp.argmax(paddle_columns, axis=-1)
    ball_x = jnp.argmax(ball_columns, axis=-1)
    return jnp.where(paddle_x < ball_x, 2, jnp.where(paddle_x > ball_x, 1, 0))


def make_evaluator(eval_episodes: int, max_steps: int):
    env, env_params = gymnax.make("Breakout-MinAtar")
    reset_env = jax.vmap(env.reset, in_axes=(0, None))
    step_env = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    @jax.jit
    def run_eval(seed: int) -> tuple[jax.Array, jax.Array]:
        key = jax.random.key(seed)
        reset_key, step_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, eval_episodes)
        obs, env_state = reset_env(reset_keys, env_params)
        initial_carry = {
            "obs": obs,
            "env_state": env_state,
            "returns": jnp.zeros((eval_episodes,)),
            "lengths": jnp.zeros((eval_episodes,)),
            "done": jnp.zeros((eval_episodes,), dtype=bool),
            "key": step_key,
        }

        def eval_step(carry: dict[str, Any], _: None):
            actions = expert_actions(carry["obs"])
            key, step_key = jax.random.split(carry["key"])
            step_keys = jax.random.split(step_key, eval_episodes)
            next_obs, next_env_state, rewards, dones, _ = step_env(
                step_keys,
                carry["env_state"],
                actions,
                env_params,
            )
            active = jnp.logical_not(carry["done"])
            returns = carry["returns"] + jnp.where(active, rewards, 0.0)
            lengths = carry["lengths"] + active.astype(jnp.float32)
            done = jnp.logical_or(carry["done"], dones)
            return {
                "obs": next_obs,
                "env_state": next_env_state,
                "returns": returns,
                "lengths": lengths,
                "done": done,
                "key": key,
            }, None

        final_carry, _ = jax.lax.scan(eval_step, initial_carry, None, max_steps)
        return final_carry["returns"], final_carry["lengths"]

    return run_eval


def summarize(returns: np.ndarray, lengths: np.ndarray) -> dict[str, float]:
    return {
        "mean_return": float(np.mean(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "median_return": float(np.median(returns)),
        "mean_length": float(np.mean(lengths)),
        "episodes_at_initial_clear": float(np.mean(returns >= 30.0)),
    }


def render_frame(obs: np.ndarray, step: int, total_return: float) -> Image.Image:
    cell = 32
    margin_top = 52
    width = cell * 10
    height = margin_top + cell * 10
    image = Image.new("RGB", (width, height), (245, 245, 242))
    draw = ImageDraw.Draw(image)
    draw.text((10, 8), f"Breakout-MinAtar expert  step={step}", fill=(20, 20, 20))
    draw.text((10, 28), f"return={total_return:.0f}", fill=(60, 60, 60))

    colors = {
        "grid": (218, 218, 214),
        "brick": (206, 72, 55),
        "trail": (114, 150, 205),
        "ball": (27, 27, 27),
        "paddle": (47, 121, 76),
    }
    for y in range(10):
        for x in range(10):
            left = x * cell
            top = margin_top + y * cell
            rect = (left, top, left + cell - 1, top + cell - 1)
            fill = (250, 250, 248)
            if obs[y, x, 3] > 0:
                fill = colors["brick"]
            if obs[y, x, 2] > 0:
                fill = colors["trail"]
            if obs[y, x, 1] > 0:
                fill = colors["ball"]
            if obs[y, x, 0] > 0:
                fill = colors["paddle"]
            draw.rectangle(rect, fill=fill, outline=colors["grid"])
    return image


def render_rollout(seed: int, max_steps: int, output_path: Path) -> dict[str, float | int | bool | str]:
    env, env_params = gymnax.make("Breakout-MinAtar")
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    obs, env_state = env.reset(reset_key, env_params)
    frames: list[Image.Image] = []
    total_return = 0.0
    done = False

    for step in range(1, max_steps + 1):
        frames.append(render_frame(np.asarray(obs), step, total_return))
        action = int(np.asarray(expert_actions(obs[None, ...]))[0])
        key, step_key = jax.random.split(key)
        obs, env_state, reward, done_value, _ = env.step(step_key, env_state, action, env_params)
        total_return += float(np.asarray(reward))
        done = bool(np.asarray(done_value))
        if done:
            frames.append(render_frame(np.asarray(obs), step + 1, total_return))
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=45, loop=0)
    return {
        "seed": seed,
        "return": total_return,
        "length": len(frames),
        "done": done,
        "gif": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve Breakout-MinAtar with an observation policy.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--eval-max-steps", type=int, default=1000)
    parser.add_argument("--target-return", type=float, default=30.0)
    parser.add_argument("--gif-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/breakout_minatar"))
    parser.add_argument("--view", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eval = make_evaluator(args.eval_episodes, args.eval_max_steps)
    returns, lengths = run_eval(args.seed)
    returns_np = np.asarray(returns)
    lengths_np = np.asarray(lengths)
    summary = summarize(returns_np, lengths_np)
    passed = summary["mean_return"] >= args.target_return

    gif_path = args.output_dir / "breakout_minatar.gif"
    rollout = render_rollout(args.gif_seed, args.eval_max_steps, gif_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "config": {
            "seed": args.seed,
            "eval_episodes": args.eval_episodes,
            "eval_max_steps": args.eval_max_steps,
            "target_return": args.target_return,
            "output_dir": str(args.output_dir),
        },
        "summary": summary,
        "passed": passed,
        "rollout": rollout,
    }
    metrics_path = args.output_dir / "solver_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"Breakout-MinAtar mean_return={summary['mean_return']:.1f} "
        f"min={summary['min_return']:.1f} max={summary['max_return']:.1f} "
        f"initial_clear_rate={summary['episodes_at_initial_clear']:.2%}",
        flush=True,
    )
    print(
        f"target_return={args.target_return:.1f} passed={passed}",
        flush=True,
    )
    print(
        f"rollout seed={rollout['seed']} return={rollout['return']:.1f} "
        f"length={rollout['length']} done={rollout['done']}",
        flush=True,
    )
    print(f"wrote metrics: {metrics_path}", flush=True)
    print(f"wrote gif: {gif_path}", flush=True)
    if args.view:
        Image.open(gif_path).show()


if __name__ == "__main__":
    main()
