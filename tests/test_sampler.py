import jax
from jax.config import config
config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(42)
import jax.numpy as jnp

import haiku as hk
from sampler import make_autoregressive_sampler

def transformer(M):
    from autoregressive import Transformer

    num_layers, model_size, num_heads = 2, 32, 4
    hidden_size = 48

    def forward_fn(x):
        model = Transformer(M, num_layers, model_size, num_heads, hidden_size)
        return model(x[..., None])

    van = hk.transform(forward_fn)
    return van

def test_shapes():
    n, num_states = 13, 40
    van = transformer(num_states)
    dummy_state_idx = jnp.arange(n, dtype=jnp.float64)
    params = van.init(key, dummy_state_idx)

    sampler, log_prob = make_autoregressive_sampler(van, n, num_states)
    batch = 200
    state_indices = sampler(params, key, batch)
    print("state_indices:", state_indices, "\nstate_indices.shape:", state_indices.shape)
    assert state_indices.shape == (batch, n)
    assert jnp.alltrue(state_indices < num_states)
    assert jnp.alltrue(state_indices[:, 1:] > state_indices[:, :-1])

    logp = log_prob(params, state_indices)
    print("logp:", logp, "\nlogp.shape:", logp.shape)
    assert logp.shape == (batch,)

def test_normalization():
    """
        Check probability normalization of the autoregressive model. Note this is a
    VERY STRONG CHECK of autoregressive property of the probability distribution.
    """
    import itertools

    n, num_states = 4, 10
    van = transformer(num_states)
    dummy_state_idx = jnp.arange(n, dtype=jnp.float64)
    params = van.init(key, dummy_state_idx)

    state_indices = jnp.array( list(itertools.combinations(range(num_states), n)) )
    print("state_indices:", state_indices, "\nstate_indices.shape:", state_indices.shape)
    assert jnp.alltrue(state_indices[:, 1:] > state_indices[:, :-1])

    _, log_prob = make_autoregressive_sampler(van, n, num_states)

    logp = log_prob(params, state_indices)
    norm = jnp.exp(logp).sum()
    print("logp:", logp, "\nnorm:", norm)
    assert jnp.allclose(norm, 1.)
