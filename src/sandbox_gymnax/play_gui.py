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
from PIL import Image

from sandbox_gymnax.play_acrobot import draw_frame as draw_acrobot_frame
from sandbox_gymnax.play_cartpole import draw_frame as draw_cartpole_frame
from sandbox_gymnax.solve_breakout_minatar import expert_actions, render_frame as draw_breakout_frame
from sandbox_gymnax.train import policy_logits


PyTree = Any


def load_params(path: Path) -> PyTree:
    with path.open("rb") as file:
        return pickle.load(file)


def greedy_policy_action(params: PyTree, obs: jax.Array) -> int:
    logits = policy_logits(params, obs[None, ...])
    return int(np.asarray(jnp.argmax(logits, axis=-1))[0])


def breakout_expert_action(obs: jax.Array) -> int:
    return int(np.asarray(expert_actions(obs[None, ...]))[0])


def default_params_path(game: str) -> Path | None:
    if game == "cartpole":
        return Path("outputs/cartpole/params.pkl")
    if game == "acrobot":
        return Path("outputs/acrobot/params.pkl")
    return None


def default_output_dir(game: str) -> Path:
    return Path("outputs") / f"{game}_gui"


def reset_frames_dir(frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frames_dir.glob("frame_*.png"):
        frame_path.unlink()


def save_frame(image: Image.Image, frames_dir: Path, index: int) -> str:
    filename = f"frame_{index:04d}.png"
    image.save(frames_dir / filename)
    return f"frames/{filename}"


def rollout_cartpole(params: PyTree, seed: int, max_steps: int, frames_dir: Path) -> list[dict[str, Any]]:
    env, env_params = gymnax.make("CartPole-v1")
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    obs, env_state = env.reset(reset_key, env_params)
    total_return = 0.0
    frames = []

    for step in range(1, max_steps + 1):
        action = greedy_policy_action(params, obs)
        image = draw_cartpole_frame(env_state, env_params, total_return, step)
        key, step_key = jax.random.split(key)
        obs, env_state, reward, done_value, _ = env.step(step_key, env_state, action, env_params)
        total_return += float(np.asarray(reward))
        done = bool(np.asarray(done_value))
        frames.append(
            {
                "src": save_frame(image, frames_dir, step - 1),
                "step": step,
                "action": action,
                "return": total_return,
                "done": done,
            }
        )
        if done:
            break
    return frames


def rollout_acrobot(params: PyTree, seed: int, max_steps: int, frames_dir: Path) -> list[dict[str, Any]]:
    env, env_params = gymnax.make("Acrobot-v1")
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    obs, env_state = env.reset(reset_key, env_params)
    total_return = 0.0
    frames = []

    for step in range(1, max_steps + 1):
        action = greedy_policy_action(params, obs)
        key, step_key = jax.random.split(key)
        next_obs, next_env_state, reward, done_value, _ = env.step(
            step_key,
            env_state,
            action,
            env_params,
        )
        total_return += float(np.asarray(reward))
        done = bool(np.asarray(done_value))
        image = draw_acrobot_frame(env_state, env_params, action, total_return, step, done)
        frames.append(
            {
                "src": save_frame(image, frames_dir, step - 1),
                "step": step,
                "action": action,
                "return": total_return,
                "done": done,
            }
        )
        obs = next_obs
        env_state = next_env_state
        if done:
            break
    return frames


def rollout_breakout(seed: int, max_steps: int, frames_dir: Path) -> list[dict[str, Any]]:
    env, env_params = gymnax.make("Breakout-MinAtar")
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    obs, env_state = env.reset(reset_key, env_params)
    total_return = 0.0
    frames = []

    for step in range(1, max_steps + 1):
        action = breakout_expert_action(obs)
        image = draw_breakout_frame(np.asarray(obs), step, total_return)
        key, step_key = jax.random.split(key)
        obs, env_state, reward, done_value, _ = env.step(step_key, env_state, action, env_params)
        total_return += float(np.asarray(reward))
        done = bool(np.asarray(done_value))
        frames.append(
            {
                "src": save_frame(image, frames_dir, step - 1),
                "step": step,
                "action": action,
                "return": total_return,
                "done": done,
            }
        )
        if done:
            break
    return frames


def write_player_html(
    output_path: Path,
    game_label: str,
    seed: int,
    fps: int,
    frames: list[dict[str, Any]],
) -> None:
    frame_json = json.dumps(frames)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{game_label} Player</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f5f2;
      color: #181a1b;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 16px;
      border-bottom: 1px solid #d6d8d2;
      background: #ffffff;
    }}
    h1 {{
      font-size: 16px;
      line-height: 1.2;
      margin: 0;
      font-weight: 650;
      letter-spacing: 0;
    }}
    main {{
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      min-height: 0;
    }}
    .stage {{
      display: grid;
      place-items: center;
      min-height: 0;
      padding: 16px;
    }}
    #frame {{
      max-width: min(100%, 920px);
      max-height: calc(100vh - 148px);
      width: auto;
      height: auto;
      image-rendering: auto;
      border: 1px solid #cfd2ca;
      background: #fff;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto auto auto auto minmax(180px, 1fr) auto;
      align-items: center;
      gap: 10px;
      padding: 12px 16px;
      border-top: 1px solid #d6d8d2;
      background: #ffffff;
    }}
    button {{
      min-width: 72px;
      height: 34px;
      border: 1px solid #b8bdb4;
      border-radius: 6px;
      background: #f9faf7;
      color: #161817;
      font: inherit;
      cursor: pointer;
    }}
    button:hover {{
      background: #eef1ea;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    .meta {{
      font-size: 13px;
      line-height: 1.4;
      color: #3b3f3a;
      white-space: nowrap;
    }}
    @media (max-width: 760px) {{
      .controls {{
        grid-template-columns: repeat(4, auto);
      }}
      .controls input,
      .controls .meta:last-child {{
        grid-column: 1 / -1;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{game_label}</h1>
    <div class="meta">seed={seed} frames={len(frames)}</div>
  </header>
  <main>
    <div class="stage">
      <img id="frame" alt="{game_label} frame">
    </div>
    <div class="controls">
      <button id="play">Play</button>
      <button id="prev">Prev</button>
      <button id="next">Next</button>
      <button id="reset">Reset</button>
      <input id="scrub" type="range" min="0" max="{max(0, len(frames) - 1)}" value="0">
      <div id="status" class="meta"></div>
    </div>
  </main>
  <script>
    const frames = {frame_json};
    let index = 0;
    let playing = false;
    let timer = null;
    let intervalMs = {max(1, int(1000 / fps))};

    const img = document.getElementById("frame");
    const playButton = document.getElementById("play");
    const prevButton = document.getElementById("prev");
    const nextButton = document.getElementById("next");
    const resetButton = document.getElementById("reset");
    const scrub = document.getElementById("scrub");
    const status = document.getElementById("status");

    function setFrame(nextIndex) {{
      index = Math.max(0, Math.min(frames.length - 1, nextIndex));
      const frame = frames[index];
      img.src = frame.src;
      scrub.value = String(index);
      status.textContent = `frame ${{index + 1}}/${{frames.length}}  step=${{frame.step}}  action=${{frame.action}}  return=${{Number(frame.return).toFixed(1)}}  done=${{frame.done}}`;
    }}

    function stop() {{
      playing = false;
      playButton.textContent = "Play";
      if (timer !== null) {{
        window.clearInterval(timer);
        timer = null;
      }}
    }}

    function play() {{
      if (playing) {{
        stop();
        return;
      }}
      playing = true;
      playButton.textContent = "Pause";
      timer = window.setInterval(() => {{
        if (index >= frames.length - 1) {{
          stop();
        }} else {{
          setFrame(index + 1);
        }}
      }}, intervalMs);
    }}

    playButton.addEventListener("click", play);
    prevButton.addEventListener("click", () => {{ stop(); setFrame(index - 1); }});
    nextButton.addEventListener("click", () => {{ stop(); setFrame(index + 1); }});
    resetButton.addEventListener("click", () => {{ stop(); setFrame(0); }});
    scrub.addEventListener("input", () => {{ stop(); setFrame(Number(scrub.value)); }});
    window.addEventListener("keydown", (event) => {{
      if (event.key === " ") {{ event.preventDefault(); play(); }}
      if (event.key === "ArrowRight") {{ stop(); setFrame(index + 1); }}
      if (event.key === "ArrowLeft") {{ stop(); setFrame(index - 1); }}
      if (event.key.toLowerCase() === "r") {{ stop(); setFrame(0); }}
    }});

    setFrame(0);
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a browser GUI for policy playback.")
    parser.add_argument("--game", choices=["cartpole", "acrobot", "breakout"], default="breakout")
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.game)
    frames_dir = output_dir / "frames"
    reset_frames_dir(frames_dir)

    if args.game == "cartpole":
        params_path = args.params or default_params_path(args.game)
        if params_path is None:
            raise ValueError("cartpole playback requires --params")
        frames = rollout_cartpole(load_params(params_path), args.seed, args.max_steps, frames_dir)
        game_label = "CartPole-v1 Policy"
    elif args.game == "acrobot":
        params_path = args.params or default_params_path(args.game)
        if params_path is None:
            raise ValueError("acrobot playback requires --params")
        frames = rollout_acrobot(load_params(params_path), args.seed, args.max_steps, frames_dir)
        game_label = "Acrobot-v1 Policy"
    else:
        frames = rollout_breakout(args.seed, args.max_steps, frames_dir)
        game_label = "Breakout-MinAtar Solver"

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "player.html"
    write_player_html(html_path, game_label, args.seed, args.fps, frames)

    last_frame = frames[-1]
    print(
        f"wrote {html_path} frames={len(frames)} return={last_frame['return']:.1f} "
        f"done={last_frame['done']}",
        flush=True,
    )
    if args.open:
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
