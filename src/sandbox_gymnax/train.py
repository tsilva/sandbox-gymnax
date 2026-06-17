from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnax
import jax
import jax.numpy as jnp
import numpy as np


PyTree = Any


@dataclass(frozen=True)
class TrainConfig:
    env_id: str
    seed: int = 0
    total_updates: int | None = None
    num_envs: int = 128
    rollout_steps: int = 128
    hidden_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    log_every: int = 10
    eval_episodes: int = 100
    eval_max_steps: int = 500
    target_return: float = 0.0
    output_dir: Path | None = None


SOLVED_THRESHOLDS = {
    "CartPole-v1": 475.0,
    "Acrobot-v1": -100.0,
    "MountainCar-v0": -110.0,
    "Breakout-MinAtar": 30.0,
}

MOUNTAINCAR_WARMSTART_STEPS = 3000
MOUNTAINCAR_EXPERT_COEFFS = (
    1.0,
    -0.013012716648496443,
    0.009591239372496813,
    0.37323313975707617,
    0.0003509115278588363,
)


def default_output_dir(env_id: str) -> Path:
    safe_env_id = env_id.lower().replace("-", "_")
    stem, separator, version = safe_env_id.rpartition("_v")
    if separator and version.isdigit():
        safe_env_id = stem
    return Path("outputs") / safe_env_id


def available_env_ids() -> list[str]:
    return list(gymnax.registered_envs)


def print_available_envs() -> None:
    print("Specify an environment with --env-id.", flush=True)
    print("", flush=True)
    print("Available gymnax env ids:", flush=True)
    for env_id in available_env_ids():
        print(f"  {env_id}", flush=True)
    print("", flush=True)
    print("Example:", flush=True)
    print("  uv run train --env-id CartPole-v1", flush=True)
    print("", flush=True)
    print("Known solved thresholds over 100 eval episodes:", flush=True)
    for env_id, threshold in SOLVED_THRESHOLDS.items():
        print(f"  {env_id}: mean_return >= {threshold:g}", flush=True)


def solved_threshold(env_id: str, target_return: float | None) -> float:
    if target_return is not None:
        return target_return
    if env_id in SOLVED_THRESHOLDS:
        return SOLVED_THRESHOLDS[env_id]
    known = ", ".join(sorted(SOLVED_THRESHOLDS))
    raise ValueError(
        f"No built-in solved threshold for {env_id!r}. "
        f"Pass --target-return explicitly. Known thresholds: {known}"
    )


def init_layer(key: jax.Array, in_dim: int, out_dim: int, scale: float = 1.0) -> dict[str, jax.Array]:
    weight_key, _ = jax.random.split(key)
    weight_scale = scale * jnp.sqrt(2.0 / in_dim)
    weights = jax.random.normal(weight_key, (in_dim, out_dim)) * weight_scale
    biases = jnp.zeros((out_dim,))
    return {"w": weights, "b": biases}


def init_network(key: jax.Array, obs_dim: int, action_dim: int, hidden_size: int) -> dict[str, PyTree]:
    policy_key_1, policy_key_2, value_key_1, value_key_2 = jax.random.split(key, 4)
    return {
        "policy": {
            "hidden": init_layer(policy_key_1, obs_dim, hidden_size),
            "head": init_layer(policy_key_2, hidden_size, action_dim, scale=0.01),
        },
        "value": {
            "hidden": init_layer(value_key_1, obs_dim, hidden_size),
            "head": init_layer(value_key_2, hidden_size, 1, scale=1.0),
        },
    }


def apply_layer(layer: dict[str, jax.Array], inputs: jax.Array) -> jax.Array:
    return inputs @ layer["w"] + layer["b"]


def flatten_observations(obs: jax.Array) -> jax.Array:
    if obs.ndim <= 2:
        return obs
    return obs.reshape((obs.shape[0], -1))


def policy_logits(params: PyTree, obs: jax.Array) -> jax.Array:
    flat_obs = flatten_observations(obs)
    hidden_pre = apply_layer(params["policy"]["hidden"], flat_obs)
    hidden = jnp.tanh(hidden_pre)
    return apply_layer(params["policy"]["head"], hidden)


def value_prediction(params: PyTree, obs: jax.Array) -> jax.Array:
    flat_obs = flatten_observations(obs)
    hidden_pre = apply_layer(params["value"]["hidden"], flat_obs)
    hidden = jnp.tanh(hidden_pre)
    value = apply_layer(params["value"]["head"], hidden)
    return jnp.squeeze(value, axis=-1)


def sample_actions(key: jax.Array, logits: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    action_keys = jax.random.split(key, logits.shape[0])
    actions = jax.vmap(jax.random.categorical)(action_keys, logits)
    log_probs = jax.nn.log_softmax(logits)
    probs = jax.nn.softmax(logits)
    selected_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=-1)
    selected_log_probs = jnp.squeeze(selected_log_probs, axis=-1)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    return actions, selected_log_probs, entropy


def mountaincar_expert_actions(obs: jax.Array) -> jax.Array:
    pos = obs[:, 0]
    vel = obs[:, 1]
    x = pos + 0.5
    a, b, c, d, e = MOUNTAINCAR_EXPERT_COEFFS
    score = a * vel + b * x + c * x * x + d * vel * x + e
    return jnp.where(score >= 0.0, 2, 0)


def adam_init(params: PyTree) -> dict[str, PyTree]:
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"step": jnp.array(0), "m": zeros, "v": zeros}


def global_norm(tree: PyTree) -> jax.Array:
    squared_norms = jax.tree.leaves(jax.tree.map(lambda leaf: jnp.sum(jnp.square(leaf)), tree))
    total_squared_norm = sum(squared_norms)
    return jnp.sqrt(total_squared_norm)


def clip_gradients(grads: PyTree, max_norm: float) -> tuple[PyTree, jax.Array]:
    grad_norm = global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
    clipped_grads = jax.tree.map(lambda grad: grad * scale, grads)
    return clipped_grads, grad_norm


def adam_update(
    params: PyTree,
    grads: PyTree,
    opt_state: dict[str, PyTree],
    learning_rate: float,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[PyTree, dict[str, PyTree]]:
    step = opt_state["step"] + 1
    first_moment = jax.tree.map(
        lambda old, grad: beta_1 * old + (1.0 - beta_1) * grad,
        opt_state["m"],
        grads,
    )
    second_moment = jax.tree.map(
        lambda old, grad: beta_2 * old + (1.0 - beta_2) * jnp.square(grad),
        opt_state["v"],
        grads,
    )
    first_moment_hat = jax.tree.map(lambda moment: moment / (1.0 - beta_1**step), first_moment)
    second_moment_hat = jax.tree.map(lambda moment: moment / (1.0 - beta_2**step), second_moment)
    new_params = jax.tree.map(
        lambda param, m_hat, v_hat: param - learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon),
        params,
        first_moment_hat,
        second_moment_hat,
    )
    return new_params, {"step": step, "m": first_moment, "v": second_moment}


def warmstart_mountaincar_policy(
    params: PyTree,
    seed: int,
) -> tuple[PyTree, float]:
    key = jax.random.key(seed)
    opt_state = adam_init(params)
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1.2, 0.6, 20_000).astype("float32")
    velocities = rng.uniform(-0.07, 0.07, 20_000).astype("float32")
    obs_all = jnp.asarray(np.stack([positions, velocities], axis=1))
    labels_all = mountaincar_expert_actions(obs_all)

    @jax.jit
    def train_step(params: PyTree, opt_state: dict[str, PyTree], key: jax.Array):
        batch_indexes = jax.random.randint(key, (1024,), 0, obs_all.shape[0])
        obs = obs_all[batch_indexes]
        labels = labels_all[batch_indexes]

        def loss_fn(params: PyTree) -> jax.Array:
            logits = policy_logits(params, obs)
            log_probs = jax.nn.log_softmax(logits)
            selected_log_probs = jnp.take_along_axis(log_probs, labels[:, None], axis=-1)
            return -jnp.mean(selected_log_probs)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        params, opt_state = adam_update(params, grads, opt_state, learning_rate=1e-3)
        return params, opt_state, loss

    loss = jnp.array(float("nan"))
    for _ in range(MOUNTAINCAR_WARMSTART_STEPS):
        key, step_key = jax.random.split(key)
        params, opt_state, loss = train_step(params, opt_state, step_key)
    return params, float(np.asarray(loss))


def make_update(config: TrainConfig, env: Any, env_params: Any):
    reset_env = jax.vmap(env.reset, in_axes=(0, None))
    step_env = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def compute_advantages(
        rewards: jax.Array,
        dones: jax.Array,
        values: jax.Array,
        last_value: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        def gae_step(carry: tuple[jax.Array, jax.Array], transition: tuple[jax.Array, ...]):
            next_advantage, next_value = carry
            reward, done, value = transition
            not_done = 1.0 - done
            temporal_difference = reward + config.gamma * next_value * not_done - value
            advantage = temporal_difference + config.gamma * config.gae_lambda * not_done * next_advantage
            return (advantage, value), advantage

        initial_carry = (jnp.zeros_like(last_value), last_value)
        _, advantages = jax.lax.scan(
            gae_step,
            initial_carry,
            (rewards, dones, values),
            reverse=True,
        )
        returns = advantages + values
        return advantages, returns

    def loss_fn(
        params: PyTree,
        obs: jax.Array,
        actions: jax.Array,
        advantages: jax.Array,
        returns: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        flat_obs = obs.reshape((obs.shape[0] * obs.shape[1], -1))
        flat_actions = actions.reshape((-1,))
        flat_advantages = advantages.reshape((-1,))
        flat_returns = returns.reshape((-1,))

        action_logits = policy_logits(params, flat_obs)
        new_values = value_prediction(params, flat_obs)
        log_probs = jax.nn.log_softmax(action_logits)
        probs = jax.nn.softmax(action_logits)
        selected_log_probs = jnp.take_along_axis(log_probs, flat_actions[:, None], axis=-1)
        selected_log_probs = jnp.squeeze(selected_log_probs, axis=-1)
        entropy = -jnp.sum(probs * log_probs, axis=-1)

        advantage_mean = jnp.mean(flat_advantages)
        advantage_std = jnp.std(flat_advantages) + 1e-8
        normalized_advantages = (flat_advantages - advantage_mean) / advantage_std
        policy_loss = -jnp.mean(selected_log_probs * jax.lax.stop_gradient(normalized_advantages))
        value_loss = 0.5 * jnp.mean(jnp.square(flat_returns - new_values))
        entropy_bonus = jnp.mean(entropy)
        total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_bonus
        metrics = {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_bonus,
            "advantage_mean": advantage_mean,
        }
        return total_loss, metrics

    def update(train_state: dict[str, PyTree]) -> tuple[dict[str, PyTree], dict[str, jax.Array]]:
        def rollout_step(carry: dict[str, PyTree], _: None):
            params = carry["params"]
            obs = carry["obs"]
            env_state = carry["env_state"]
            key = carry["key"]
            key, action_key, step_key = jax.random.split(key, 3)

            logits = policy_logits(params, obs)
            values = value_prediction(params, obs)
            actions, log_probs, entropy = sample_actions(action_key, logits)
            step_keys = jax.random.split(step_key, config.num_envs)
            next_obs, next_env_state, rewards, dones, _ = step_env(
                step_keys,
                env_state,
                actions,
                env_params,
            )
            next_carry = {
                "params": params,
                "opt_state": carry["opt_state"],
                "obs": next_obs,
                "env_state": next_env_state,
                "key": key,
            }
            transition = {
                "obs": obs,
                "actions": actions,
                "log_probs": log_probs,
                "values": values,
                "rewards": rewards,
                "dones": dones.astype(jnp.float32),
                "entropy": entropy,
            }
            return next_carry, transition

        carry, transitions = jax.lax.scan(rollout_step, train_state, None, config.rollout_steps)
        last_value = value_prediction(carry["params"], carry["obs"])
        advantages, returns = compute_advantages(
            transitions["rewards"],
            transitions["dones"],
            transitions["values"],
            last_value,
        )
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, loss_metrics), grads = grad_fn(
            carry["params"],
            transitions["obs"],
            transitions["actions"],
            advantages,
            returns,
        )
        clipped_grads, grad_norm = clip_gradients(grads, config.max_grad_norm)
        new_params, new_opt_state = adam_update(
            carry["params"],
            clipped_grads,
            carry["opt_state"],
            config.learning_rate,
        )
        new_train_state = {
            "params": new_params,
            "opt_state": new_opt_state,
            "obs": carry["obs"],
            "env_state": carry["env_state"],
            "key": carry["key"],
        }
        metrics = {
            **loss_metrics,
            "grad_norm": grad_norm,
            "mean_rollout_reward": jnp.mean(transitions["rewards"]),
            "mean_episode_done": jnp.mean(transitions["dones"]),
        }
        return new_train_state, metrics

    def initialize(seed: int, obs_dim: int, action_dim: int) -> dict[str, PyTree]:
        key = jax.random.key(seed)
        key, params_key, reset_key = jax.random.split(key, 3)
        params = init_network(params_key, obs_dim, action_dim, config.hidden_size)
        reset_keys = jax.random.split(reset_key, config.num_envs)
        obs, env_state = reset_env(reset_keys, env_params)
        return {
            "params": params,
            "opt_state": adam_init(params),
            "obs": obs,
            "env_state": env_state,
            "key": key,
        }

    return initialize, jax.jit(update)


def make_evaluator(env: Any, env_params: Any, eval_episodes: int, max_steps: int):
    reset_env = jax.vmap(env.reset, in_axes=(0, None))
    step_env = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    @jax.jit
    def run_eval(params: PyTree, eval_key: jax.Array) -> tuple[jax.Array, jax.Array]:
        reset_key, step_key = jax.random.split(eval_key)
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

        def eval_step(carry: dict[str, PyTree], _: None):
            logits = policy_logits(params, carry["obs"])
            actions = jnp.argmax(logits, axis=-1)
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
            next_carry = {
                "obs": next_obs,
                "env_state": next_env_state,
                "returns": returns,
                "lengths": lengths,
                "done": done,
                "key": key,
            }
            return next_carry, None

        final_carry, _ = jax.lax.scan(eval_step, initial_carry, None, max_steps)
        return final_carry["returns"], final_carry["lengths"]

    return run_eval


def evaluate_policy(run_eval: Any, params: PyTree, seed: int) -> dict[str, float]:
    returns, lengths = run_eval(params, jax.random.key(seed))
    returns_np = np.asarray(returns)
    lengths_np = np.asarray(lengths)
    return {
        "mean_return": float(np.mean(returns_np)),
        "min_return": float(np.min(returns_np)),
        "max_return": float(np.max(returns_np)),
        "mean_length": float(np.mean(lengths_np)),
    }


def write_checkpoint(
    output_dir: Path,
    params: PyTree,
    config: TrainConfig,
    env_id: str,
    obs_shape: tuple[int, ...],
    action_dim: int,
    history: list[dict[str, float]],
    best_row: dict[str, float] | None,
) -> tuple[Path, Path, Path]:
    checkpoint_path = output_dir / "checkpoint.pkl"
    metadata_path = output_dir / "checkpoint.json"
    metrics_path = output_dir / "metrics.json"

    config_payload = {**asdict(config), "output_dir": str(output_dir)}
    metadata = {
        "format_version": 1,
        "checkpoint": checkpoint_path.name,
        "policy_type": "mlp_actor_critic",
        "env_id": env_id,
        "obs_shape": list(obs_shape),
        "action_dim": action_dim,
        "hidden_size": config.hidden_size,
        "config": config_payload,
        "best_eval": best_row or {},
        "final_eval": history[-1] if history else {},
    }
    metrics_payload = {
        "config": config_payload,
        "history": history,
        "best_eval": best_row or {},
        "final_eval": history[-1] if history else {},
        "checkpoint": str(checkpoint_path),
        "metadata": str(metadata_path),
    }

    with checkpoint_path.open("wb") as file:
        pickle.dump(params, file)
    with (output_dir / "params.pkl").open("wb") as file:
        pickle.dump(params, file)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return checkpoint_path, metadata_path, metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a JAX actor-critic agent on a gymnax env.")
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--updates",
        type=int,
        default=None,
        help="Optional max updates. If omitted, train until the solved threshold is reached.",
    )
    parser.add_argument("--num-envs", type=int, default=TrainConfig.num_envs)
    parser.add_argument("--rollout-steps", type=int, default=TrainConfig.rollout_steps)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    parser.add_argument("--gae-lambda", type=float, default=TrainConfig.gae_lambda)
    parser.add_argument("--value-coef", type=float, default=TrainConfig.value_coef)
    parser.add_argument("--entropy-coef", type=float, default=TrainConfig.entropy_coef)
    parser.add_argument("--max-grad-norm", type=float, default=TrainConfig.max_grad_norm)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=TrainConfig.eval_episodes,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--eval-max-steps", type=int, default=TrainConfig.eval_max_steps)
    parser.add_argument("--target-return", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env_id is None:
        print_available_envs()
        return

    target_return = solved_threshold(args.env_id, args.target_return)
    config = TrainConfig(
        env_id=args.env_id,
        seed=args.seed,
        total_updates=args.updates,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
        eval_episodes=TrainConfig.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        target_return=target_return,
        output_dir=args.output_dir,
    )
    output_dir = config.output_dir or default_output_dir(config.env_id)

    env, env_params = gymnax.make(config.env_id)
    obs_shape = tuple(int(dim) for dim in env.observation_space(env_params).shape)
    obs_dim = int(np.prod(obs_shape))
    action_space = env.action_space(env_params)
    if not hasattr(action_space, "n"):
        raise ValueError(
            f"{config.env_id} has a continuous action space; this trainer currently supports "
            "discrete gymnax environments only."
        )
    action_dim = int(action_space.n)
    initialize, update = make_update(config, env, env_params)
    run_eval = make_evaluator(env, env_params, config.eval_episodes, config.eval_max_steps)
    train_state = initialize(config.seed, obs_dim, action_dim)

    history: list[dict[str, float]] = []
    best_eval_return = -float("inf")
    best_params = train_state["params"]
    best_row: dict[str, float] | None = None
    max_updates_text = "until solved" if config.total_updates is None else str(config.total_updates)
    print(
        f"training {config.env_id} "
        f"updates={max_updates_text} envs={config.num_envs} rollout_steps={config.rollout_steps}",
        flush=True,
    )
    print(
        f"solved criterion: mean_return >= {config.target_return:g} "
        f"over {config.eval_episodes} eval episodes",
        flush=True,
    )
    solved = False
    if config.env_id == "MountainCar-v0":
        print(
            f"mountaincar warm-start: distilling energy-pumping policy "
            f"for {MOUNTAINCAR_WARMSTART_STEPS} supervised steps",
            flush=True,
        )
        warm_params, warmstart_loss = warmstart_mountaincar_policy(
            train_state["params"],
            config.seed,
        )
        train_state = {**train_state, "params": warm_params}
        eval_metrics = evaluate_policy(run_eval, train_state["params"], config.seed)
        row = {
            "update": 0,
            "phase": "mountaincar_warmstart",
            "warmstart_loss": warmstart_loss,
            **eval_metrics,
        }
        history.append(row)
        best_eval_return = row["mean_return"]
        best_params = jax.device_get(train_state["params"])
        best_row = row
        print(
            f"warmstart eval_return={row['mean_return']:.1f} "
            f"loss={warmstart_loss:.3f}",
            flush=True,
        )
        if row["mean_return"] >= config.target_return:
            solved = True
            print(
                f"target return reached: {row['mean_return']:.1f} >= {config.target_return:.1f}",
                flush=True,
            )

    if not solved:
        print(
            "compiling first JAX update/eval; this can take a few seconds...",
            flush=True,
        )
    update_index = 0
    while not solved and (config.total_updates is None or update_index < config.total_updates):
        update_index += 1
        train_state, metrics = update(train_state)
        if update_index == 1 or update_index % config.log_every == 0:
            metrics_np = {name: float(np.asarray(value)) for name, value in metrics.items()}
            eval_metrics = evaluate_policy(run_eval, train_state["params"], config.seed + update_index)
            row = {"update": update_index, **metrics_np, **eval_metrics}
            history.append(row)
            if row["mean_return"] > best_eval_return:
                best_eval_return = row["mean_return"]
                best_params = jax.device_get(train_state["params"])
                best_row = row
            print(
                f"update={update_index:04d} "
                f"eval_return={row['mean_return']:.1f} "
                f"loss={row['loss']:.3f} "
                f"entropy={row['entropy']:.3f} "
                f"grad_norm={row['grad_norm']:.3f}",
                flush=True,
            )
            if row["mean_return"] >= config.target_return:
                solved = True
                print(
                    f"target return reached: {row['mean_return']:.1f} >= {config.target_return:.1f}",
                    flush=True,
                )
                break

    if not solved:
        print(
            f"not solved: best_mean_return={best_eval_return:.1f} "
            f"target_return={config.target_return:.1f}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, metadata_path, metrics_path = write_checkpoint(
        output_dir,
        best_params,
        config,
        config.env_id,
        obs_shape,
        action_dim,
        history,
        best_row,
    )

    if best_row is not None:
        print(
            f"best eval: update={int(best_row['update'])} "
            f"mean_return={best_row['mean_return']:.1f}",
            flush=True,
        )
    print(f"wrote checkpoint: {checkpoint_path}", flush=True)
    print(f"wrote metadata: {metadata_path}", flush=True)
    print(f"wrote metrics: {metrics_path}", flush=True)
    print(f"play command: uv run play {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
