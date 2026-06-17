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
        return pickle.load(file)


def greedy_action(params: PyTree, obs: jax.Array) -> int:
    logits = policy_logits(params, obs[None, :])
    return int(np.asarray(jnp.argmax(logits, axis=-1))[0])


def rollout_policy(
    params: PyTree,
    seed: int,
    max_steps: int,
) -> tuple[list[Any], list[int], list[float], float, int, bool]:
    env, env_params = gymnax.make("Acrobot-v1")
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    obs, env_state = env.reset(reset_key, env_params)

    states: list[Any] = []
    actions: list[int] = []
    rewards: list[float] = []
    done = False
    for step in range(max_steps):
        states.append(env_state)
        action = greedy_action(params, obs)
        actions.append(action)
        key, step_key = jax.random.split(key)
        obs, env_state, reward, done_value, _ = env.step(step_key, env_state, action, env_params)
        rewards.append(float(np.asarray(reward)))
        done = bool(np.asarray(done_value))
        if done:
            return states, actions, rewards, float(np.sum(rewards)), step + 1, done

    return states, actions, rewards, float(np.sum(rewards)), max_steps, done


def draw_frame(
    state: Any,
    env_params: Any,
    action: int,
    cumulative_reward: float,
    step: int,
    done: bool,
) -> Image.Image:
    width = 520
    height = 520
    origin = np.array([width / 2, height * 0.47])
    scale = 115.0
    link_1 = float(np.asarray(env_params.link_length_1))
    link_2 = float(np.asarray(env_params.link_length_2))

    theta_1 = float(np.asarray(state.joint_angle1))
    theta_2 = float(np.asarray(state.joint_angle2))
    joint = origin + scale * link_1 * np.array([np.sin(theta_1), np.cos(theta_1)])
    tip = joint + scale * link_2 * np.array([np.sin(theta_1 + theta_2), np.cos(theta_1 + theta_2)])

    image = Image.new("RGB", (width, height), (248, 248, 245))
    draw = ImageDraw.Draw(image)
    goal_y = origin[1] - scale
    draw.line((0, goal_y, width, goal_y), fill=(84, 145, 96), width=2)
    draw.text((18, 16), f"Acrobot-v1 policy  step={step}", fill=(20, 20, 20))
    draw.text(
        (18, 38),
        f"return={cumulative_reward:.0f}  action={action}  done={done}",
        fill=(60, 60, 60),
    )
    draw.line((*origin, *joint), fill=(44, 92, 160), width=9)
    draw.line((*joint, *tip), fill=(205, 83, 52), width=9)
    for point, radius, color in [
        (origin, 9, (30, 30, 30)),
        (joint, 8, (30, 30, 30)),
        (tip, 7, (20, 110, 45)),
    ]:
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image


def save_gif(
    output: Path,
    states: list[Any],
    actions: list[int],
    rewards: list[float],
    env_params: Any,
    fps: int,
    done: bool,
) -> None:
    cumulative_rewards = np.cumsum(np.asarray(rewards, dtype=np.float32))
    frames = [
        draw_frame(
            state,
            env_params,
            actions[index],
            float(cumulative_rewards[index]),
            index + 1,
            done and index == len(states) - 1,
        )
        for index, state in enumerate(states)
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
    parser = argparse.ArgumentParser(description="Render the trained Acrobot policy.")
    parser.add_argument("--params", type=Path, default=Path("outputs/acrobot/params.pkl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/acrobot/acrobot.gif"))
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--view", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.params)
    states, actions, rewards, episode_return, episode_length, done = rollout_policy(
        params,
        args.seed,
        args.max_steps,
    )

    _, env_params = gymnax.make("Acrobot-v1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_gif(args.output, states, actions, rewards, env_params, args.fps, done)
    if args.view:
        Image.open(args.output).show()

    print(
        f"rendered {args.output} "
        f"return={episode_return:.1f} length={episode_length} done={done} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
