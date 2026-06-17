from __future__ import annotations

import argparse
import json
import pickle
import webbrowser
from pathlib import Path
from typing import Any

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from sandbox_gymnax.play_acrobot import draw_frame as draw_acrobot_frame
from sandbox_gymnax.play_cartpole import draw_frame as draw_cartpole_frame
from sandbox_gymnax.play_gui import reset_frames_dir, save_frame, write_player_html
from sandbox_gymnax.solve_breakout_minatar import render_frame as draw_breakout_frame
from sandbox_gymnax.train import policy_logits


PyTree = Any


def load_checkpoint(path: Path) -> PyTree:
    with path.open("rb") as file:
        return pickle.load(file)


def infer_metadata_path(checkpoint_path: Path, metadata_path: Path | None) -> Path:
    if metadata_path is not None:
        return metadata_path
    candidates = [
        checkpoint_path.with_suffix(".json"),
        checkpoint_path.parent / "checkpoint.json",
        checkpoint_path.parent / "metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"could not find metadata for {checkpoint_path}; expected one of "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_metadata(checkpoint_path: Path, metadata_path: Path | None) -> dict[str, Any]:
    resolved_metadata_path = infer_metadata_path(checkpoint_path, metadata_path)
    with resolved_metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if "env_id" not in metadata:
        raise ValueError(f"{resolved_metadata_path} does not contain required key 'env_id'")
    if metadata.get("policy_type") != "mlp_actor_critic":
        raise ValueError(
            f"unsupported policy_type={metadata.get('policy_type')!r}; "
            "play.py currently supports mlp_actor_critic checkpoints"
        )
    return metadata


def greedy_action(params: PyTree, obs: jax.Array) -> int:
    logits = policy_logits(params, obs[None, ...])
    return int(np.asarray(jnp.argmax(logits, axis=-1))[0])


def draw_frame(
    env_id: str,
    obs: jax.Array,
    env_state: Any,
    env_params: Any,
    action: int,
    cumulative_return: float,
    step: int,
    done: bool,
) -> Image.Image:
    if env_id == "CartPole-v1":
        return draw_cartpole_frame(env_state, env_params, cumulative_return, step)
    if env_id == "Acrobot-v1":
        return draw_acrobot_frame(env_state, env_params, action, cumulative_return, step, done)
    if env_id == "Breakout-MinAtar":
        return draw_breakout_frame(np.asarray(obs), step, cumulative_return)
    if env_id == "MountainCar-v0":
        return draw_mountaincar_frame(env_state, env_params, action, cumulative_return, step, done)
    raise ValueError(
        f"play.py does not have a renderer for {env_id!r}; supported renderers are "
        "CartPole-v1, Acrobot-v1, MountainCar-v0, and Breakout-MinAtar"
    )


def draw_mountaincar_frame(
    state: Any,
    env_params: Any,
    action: int,
    cumulative_return: float,
    step: int,
    done: bool,
) -> Image.Image:
    width = 720
    height = 420
    image = Image.new("RGB", (width, height), (248, 248, 245))
    draw = ImageDraw.Draw(image)

    margin_x = 54
    ground_top = 72
    ground_height = 250
    min_position = float(env_params.min_position)
    max_position = float(env_params.max_position)
    goal_position = float(env_params.goal_position)
    position = float(np.asarray(state.position))
    velocity = float(np.asarray(state.velocity))

    def world_to_screen_x(x: float) -> float:
        span = max_position - min_position
        return margin_x + (x - min_position) / span * (width - 2 * margin_x)

    def terrain_y(x: float) -> float:
        normalized = np.sin(3.0 * x) * 0.45 + 0.55
        return ground_top + (1.0 - normalized) * ground_height

    terrain = []
    samples = 180
    for index in range(samples + 1):
        x_world = min_position + (max_position - min_position) * index / samples
        terrain.append((world_to_screen_x(x_world), terrain_y(x_world)))
    draw.line(terrain, fill=(63, 69, 62), width=4)

    goal_x = world_to_screen_x(goal_position)
    goal_y = terrain_y(goal_position)
    draw.line((goal_x, goal_y, goal_x, goal_y - 72), fill=(38, 113, 70), width=3)
    draw.polygon(
        [(goal_x, goal_y - 72), (goal_x + 42, goal_y - 58), (goal_x, goal_y - 44)],
        fill=(42, 142, 85),
    )

    car_x = world_to_screen_x(position)
    car_y = terrain_y(position)
    car_width = 46
    car_height = 24
    car_rect = (
        car_x - car_width / 2,
        car_y - car_height - 5,
        car_x + car_width / 2,
        car_y - 5,
    )
    draw.rounded_rectangle(car_rect, radius=5, fill=(49, 102, 176), outline=(20, 28, 38), width=2)
    wheel_y = car_y - 5
    for wheel_x in (car_x - 15, car_x + 15):
        draw.ellipse((wheel_x - 6, wheel_y - 6, wheel_x + 6, wheel_y + 6), fill=(22, 24, 24))

    action_label = {0: "left", 1: "coast", 2: "right"}.get(action, str(action))
    draw.text((22, 18), f"MountainCar-v0 policy  step={step}", fill=(20, 20, 20))
    draw.text(
        (22, 42),
        (
            f"return={cumulative_return:.0f}  action={action_label}  "
            f"pos={position:.3f}  vel={velocity:.3f}  done={done}"
        ),
        fill=(58, 58, 58),
    )
    return image


def rollout_checkpoint(
    checkpoint: PyTree,
    metadata: dict[str, Any],
    seed: int,
    max_steps: int,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    env_id = str(metadata["env_id"])
    env, env_params = gymnax.make(env_id)
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    obs, env_state = env.reset(reset_key, env_params)

    images: list[Image.Image] = []
    frames: list[dict[str, Any]] = []
    cumulative_return = 0.0
    for step in range(1, max_steps + 1):
        action = greedy_action(checkpoint, obs)
        key, step_key = jax.random.split(key)
        next_obs, next_env_state, reward, done_value, _ = env.step(
            step_key,
            env_state,
            action,
            env_params,
        )
        cumulative_return += float(np.asarray(reward))
        done = bool(np.asarray(done_value))
        image = draw_frame(
            env_id,
            obs,
            env_state,
            env_params,
            action,
            cumulative_return,
            step,
            done,
        )
        images.append(image)
        frames.append(
            {
                "step": step,
                "action": action,
                "return": cumulative_return,
                "done": done,
            }
        )
        obs = next_obs
        env_state = next_env_state
        if done:
            break
    return images, frames


def write_gif(output_path: Path, images: list[Image.Image], fps: int) -> None:
    duration_ms = int(1000 / fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def write_browser_player(
    output_dir: Path,
    images: list[Image.Image],
    frames: list[dict[str, Any]],
    env_id: str,
    seed: int,
    fps: int,
) -> Path:
    frames_dir = output_dir / "frames"
    reset_frames_dir(frames_dir)
    browser_frames = []
    for index, (image, frame) in enumerate(zip(images, frames, strict=True)):
        browser_frames.append({"src": save_frame(image, frames_dir, index), **frame})
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "player.html"
    write_player_html(html_path, f"{env_id} Checkpoint", seed, fps, browser_frames)
    return html_path


def scale_to_fit(source_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[int, int, int, int]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    scale = min(target_width / source_width, target_height / source_height)
    width = max(1, int(source_width * scale))
    height = max(1, int(source_height * scale))
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    return left, top, width, height


def run_python_gui(
    images: list[Image.Image],
    frames: list[dict[str, Any]],
    env_id: str,
    checkpoint_path: Path,
    fps: int,
) -> None:
    import pygame

    pygame.init()
    try:
        pygame.display.set_caption(f"{env_id} - {checkpoint_path.name}")
        initial_size = images[0].size
        screen = pygame.display.set_mode(initial_size, pygame.RESIZABLE)
        surfaces = []
        for image in images:
            rgb_image = image.convert("RGB")
            surface = pygame.image.frombytes(rgb_image.tobytes(), rgb_image.size, "RGB").convert()
            surfaces.append(surface)

        clock = pygame.time.Clock()
        index = 0
        playing = True
        interval_ms = max(1, int(1000 / fps))
        elapsed_ms = 0
        running = True

        def render() -> None:
            screen.fill((18, 20, 22))
            surface = surfaces[index]
            left, top, width, height = scale_to_fit(surface.get_size(), screen.get_size())
            scaled = pygame.transform.smoothscale(surface, (width, height))
            screen.blit(scaled, (left, top))
            frame = frames[index]
            pygame.display.set_caption(
                f"{env_id} | frame {index + 1}/{len(frames)} "
                f"step={frame['step']} action={frame['action']} "
                f"return={frame['return']:.1f} done={frame['done']}"
            )
            pygame.display.flip()

        render()
        while running:
            delta_ms = clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        playing = not playing
                    elif event.key == pygame.K_RIGHT:
                        playing = False
                        index = min(len(frames) - 1, index + 1)
                        render()
                    elif event.key == pygame.K_LEFT:
                        playing = False
                        index = max(0, index - 1)
                        render()
                    elif event.key == pygame.K_r:
                        playing = False
                        index = 0
                        render()
                elif event.type == pygame.VIDEORESIZE:
                    render()

            if playing and running:
                elapsed_ms += delta_ms
                if elapsed_ms >= interval_ms:
                    elapsed_ms %= interval_ms
                    if index < len(frames) - 1:
                        index += 1
                        render()
                    else:
                        playing = False
    finally:
        pygame.quit()


def default_output_dir(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / "playback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained checkpoint using its metadata.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--mode", choices=["gui", "browser", "gif"], default="gui")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--open", action="store_true", help="Open browser output when --mode browser is used.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    metadata = load_metadata(args.checkpoint, args.metadata)
    images, frames = rollout_checkpoint(checkpoint, metadata, args.seed, args.max_steps)
    final_frame = frames[-1]

    if args.mode == "gui":
        print(
            f"opening Python GUI for {args.checkpoint} env={metadata['env_id']} "
            f"frames={len(frames)} return={final_frame['return']:.1f} done={final_frame['done']}",
            flush=True,
        )
        print("controls: space=pause/play, arrows=step, r=reset, q/escape=quit", flush=True)
        run_python_gui(images, frames, str(metadata["env_id"]), args.checkpoint, args.fps)
        return
    elif args.mode == "gif":
        output_path = args.output or default_output_dir(args.checkpoint) / "rollout.gif"
        write_gif(output_path, images, args.fps)
        result_path = output_path
    else:
        output_dir = args.output_dir or default_output_dir(args.checkpoint)
        result_path = write_browser_player(
            output_dir,
            images,
            frames,
            str(metadata["env_id"]),
            args.seed,
            args.fps,
        )

    print(
        f"played {args.checkpoint} env={metadata['env_id']} "
        f"frames={len(frames)} return={final_frame['return']:.1f} done={final_frame['done']}",
        flush=True,
    )
    print(f"wrote playback: {result_path}", flush=True)
    if args.open:
        webbrowser.open(result_path.resolve().as_uri())


if __name__ == "__main__":
    main()
