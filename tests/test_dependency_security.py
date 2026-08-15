"""Security-boundary and core RL regression tests."""

from importlib.metadata import version
from io import BytesIO

import gymnax
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from sandbox_gymnax.solve_breakout_minatar import expert_actions, render_frame
from sandbox_gymnax.train import init_network, policy_logits, value_prediction


def _version_tuple(distribution: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version(distribution).split(".") if part.isdigit())


def test_locked_packages_are_outside_vulnerable_ranges() -> None:
    assert _version_tuple("Pillow") >= (12, 3, 0)
    assert _version_tuple("msgpack") >= (1, 2, 1)
    assert _version_tuple("pytest") >= (9, 0, 3)


def test_pillow_accepts_rendered_frame_and_rejects_malformed_input() -> None:
    observation = np.zeros((10, 10, 4), dtype=np.float32)
    observation[9, 4, 0] = 1
    observation[5, 6, 1] = 1
    frame = render_frame(observation, step=1, total_return=0)
    buffer = BytesIO()
    frame.save(buffer, format="PNG")
    buffer.seek(0)

    with Image.open(buffer) as image:
        image.load()
        assert image.size == (320, 372)

    with pytest.raises(UnidentifiedImageError):
        Image.open(BytesIO(b"not-an-image")).load()


def test_msgpack_roundtrip_and_rejects_trailing_malicious_data() -> None:
    payload = {"returns": [1, 2, 3], "safe": True}
    packed = msgpack.packb(payload, use_bin_type=True)
    assert msgpack.unpackb(packed, raw=False) == payload

    with pytest.raises(msgpack.ExtraData):
        msgpack.unpackb(packed + msgpack.packb("unexpected"), raw=False)


def test_jax_policy_forward_backward_and_gymnax_step() -> None:
    params = init_network(jax.random.key(0), obs_dim=4, action_dim=2, hidden_size=8)
    observations = jnp.zeros((3, 4), dtype=jnp.float32)
    logits = policy_logits(params, observations)
    values = value_prediction(params, observations)
    gradients = jax.grad(lambda network: jnp.sum(policy_logits(network, observations)))(params)

    assert logits.shape == (3, 2)
    assert values.shape == (3,)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))

    env, env_params = gymnax.make("CartPole-v1")
    observation, state = env.reset(jax.random.key(1), env_params)
    next_observation, _, reward, done, _ = env.step(
        jax.random.key(2), state, jnp.array(0), env_params
    )
    assert observation.shape == (4,)
    assert next_observation.shape == (4,)
    assert jnp.isfinite(reward)
    assert done.shape == ()


def test_breakout_expert_moves_toward_ball() -> None:
    observation = jnp.zeros((1, 10, 10, 4))
    observation = observation.at[0, 9, 3, 0].set(1)
    observation = observation.at[0, 4, 7, 1].set(1)
    assert int(expert_actions(observation)[0]) == 2
