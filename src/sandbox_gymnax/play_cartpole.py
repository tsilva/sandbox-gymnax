from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from sandbox_gymnax.train import policy_logits


PyTree = Any


def load_params(path: Path) -> PyTree:
    with path.open("rb") as file:
        params = pickle.load(file)
    return params


def greedy_action(params: PyTree, obs: jax.Array) -> int:
    # Add a batch axis so the shared policy path stays explicit: (obs_dim) -> (1, obs_dim).
    batched_obs = obs[None, :]
    logits = policy_logits(params, batched_obs)
    action = jnp.argmax(logits[0], axis=-1)
    return int(action)


def rollout_policy(
    params: PyTree,
    seed: int,
    max_steps: int,
) -> tuple[list[Any], list[float], float, int]:
    env, env_params = gymnax.make("CartPole-v1")
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    obs, env_state = env.reset(reset_key, env_params)

    state_seq = []
    rewards = []
    for step in range(max_steps):
        state_seq.append(env_state)
        key, step_key = jax.random.split(key)
        action = greedy_action(params, obs)
        obs, env_state, reward, done, _ = env.step(step_key, env_state, action, env_params)
        rewards.append(float(reward))
        if bool(done):
            return state_seq, rewards, float(np.sum(rewards)), step + 1

    return state_seq, rewards, float(np.sum(rewards)), max_steps


def draw_frame(state: Any, env_params: Any, cumulative_reward: float, step: int) -> Image.Image:
    width = 720
    height = 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Project CartPole's physical x position into screen coordinates.
    track_margin = 80
    track_width = width - 2 * track_margin
    world_width = 2.0 * float(env_params.x_threshold)
    x = float(state.x)
    cart_center_x = track_margin + (x + float(env_params.x_threshold)) / world_width * track_width
    cart_center_y = 285

    # Draw the track and cart body.
    draw.line((track_margin, cart_center_y + 20, width - track_margin, cart_center_y + 20), fill="black", width=3)
    cart_width = 70
    cart_height = 34
    cart_left = cart_center_x - cart_width / 2
    cart_top = cart_center_y - cart_height / 2
    cart_right = cart_center_x + cart_width / 2
    cart_bottom = cart_center_y + cart_height / 2
    draw.rounded_rectangle(
        (cart_left, cart_top, cart_right, cart_bottom),
        radius=5,
        fill="#2f6f9f",
        outline="black",
        width=2,
    )
    draw.ellipse((cart_left + 8, cart_bottom - 3, cart_left + 24, cart_bottom + 13), fill="black")
    draw.ellipse((cart_right - 24, cart_bottom - 3, cart_right - 8, cart_bottom + 13), fill="black")

    # Draw the pole from the cart pivot. Gymnax theta is measured from vertical.
    theta = float(state.theta)
    pole_length = 150
    pivot_x = cart_center_x
    pivot_y = cart_top
    pole_tip_x = pivot_x + np.sin(theta) * pole_length
    pole_tip_y = pivot_y - np.cos(theta) * pole_length
    draw.line((pivot_x, pivot_y, pole_tip_x, pole_tip_y), fill="#b94a37", width=8)
    draw.ellipse((pivot_x - 6, pivot_y - 6, pivot_x + 6, pivot_y + 6), fill="black")
    draw.ellipse((pole_tip_x - 7, pole_tip_y - 7, pole_tip_x + 7, pole_tip_y + 7), fill="#b94a37")

    draw.text((24, 22), f"CartPole-v1 policy  step={step}", fill="black")
    draw.text((24, 48), f"return={cumulative_reward:.1f}  x={x:.2f}  theta={theta:.3f}", fill="black")
    return image


def save_gif(
    output: Path,
    state_seq: list[Any],
    rewards: list[float],
    env_params: Any,
    fps: int,
) -> None:
    cumulative_rewards = np.cumsum(np.asarray(rewards, dtype=np.float32))
    frames = [
        draw_frame(state, env_params, float(cumulative_rewards[index]), index + 1)
        for index, state in enumerate(state_seq)
    ]
    duration_ms = int(1000 / fps)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the trained CartPole policy.")
    parser.add_argument("--params", type=Path, default=Path("outputs/cartpole/params.pkl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cartpole/cartpole.gif"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--view", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.params)
    state_seq, rewards, episode_return, episode_length = rollout_policy(
        params,
        args.seed,
        args.max_steps,
    )

    _, env_params = gymnax.make("CartPole-v1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_gif(args.output, state_seq, rewards, env_params, args.fps)
    if args.view:
        Image.open(args.output).show()

    print(
        f"rendered {args.output} "
        f"return={episode_return:.1f} length={episode_length} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
