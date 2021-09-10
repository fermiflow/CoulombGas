import jax
from jax.config import config
config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(42)
import jax.numpy as jnp

import numpy as np
import haiku as hk
from autoregressive import CausalSelfAttention, Transformer
from jax.flatten_util import ravel_pytree

def test_CausalSelfAttention_params():

    """ Test parameter number of a single causal MHA layer. """

    model_size, num_heads = 32, 4
    key_size = model_size // num_heads
    init_scale = 1.0

    def forward_fn(x):
        model = CausalSelfAttention(num_heads, key_size, init_scale)
        return model(x)
    attn = hk.transform(forward_fn)

    n = 50
    x = jnp.array( np.random.randn(n, model_size) )
    params = attn.init(key, x)

    raveled_params, _ = ravel_pytree(params)
    print("Total number of parameters:", raveled_params.size)
    num_params = (model_size + 1) * key_size * 3 * num_heads \
                 + (num_heads * key_size + 1) * model_size
    print("num_params:", num_params)
    assert raveled_params.size == num_params

def test_CausalSelfAttention_autoregressive():

    """ Test autoregressive property of a single causal MHA layer. """

    model_size, num_heads = 32, 4
    key_size = model_size // num_heads
    init_scale = 1.0

    def forward_fn(x):
        model = CausalSelfAttention(num_heads, key_size, init_scale)
        return model(x)
    attn = hk.transform(forward_fn)

    n = 50
    x = jnp.array( np.random.randn(n, model_size) )
    params = attn.init(key, x)

    output = attn.apply(params, None, x)
    print("x.shape:", x.shape, "output.shape:", output.shape)
    assert output.shape == (n, model_size)

    random_vec = jnp.array( np.random.randn(n, model_size) )
    jac = jax.jacrev(lambda x: (attn.apply(params, None, x) * random_vec).sum(axis=-1))(x)
    assert jac.shape == (n, n, model_size)
    jac = jnp.linalg.norm(jac, axis=-1)
    print("jac:", jac)
    print("jac.shape:", jac.shape)

    depends = (jac != 0.).astype(int)
    assert jnp.allclose(depends, jnp.tril(jnp.ones((n, n))))

def test_Transformer_params():

    """ Test parameter number of a complete masked transformer. """

    n, M = 13, 81
    num_layers, model_size, num_heads = 2, 32, 4
    key_size = model_size // num_heads
    hidden_size = 48

    def forward_fn(x):
        model = Transformer(M, num_layers, model_size, num_heads, hidden_size)
        return model(x[..., None])
    van = hk.transform(forward_fn)

    x = jnp.array( np.random.choice(M, size=n, replace=False), dtype=jnp.float64 )
    params = van.init(key, x)

    raveled_params, _ = ravel_pytree(params)
    print("Total number of parameters:", raveled_params.size)

    embedding_mlp = (1+1) * model_size
    output_mlp = (model_size + 1) * M
    MHA = (model_size + 1) * key_size * 3 * num_heads \
                 + (num_heads * key_size + 1) * model_size
    mlp_block = (model_size + 1) * hidden_size + (hidden_size + 1) * model_size
    print("x1hat:", M, "\tembedding_mlp:", embedding_mlp, "\toutput_mlp:", output_mlp,
            "\tMHA:", MHA, "\tmlp_block:", mlp_block)
    num_params = M + embedding_mlp + output_mlp + num_layers * (MHA + mlp_block)
    print("num_params:", num_params)
    assert raveled_params.size == num_params

def test_Transformer_autoregressive():

    """ Test autoregressive property of a complete masked transformer. """

    n, M = 13, 81
    num_layers, model_size, num_heads = 2, 32, 4
    hidden_size = 48

    def forward_fn(x):
        model = Transformer(M, num_layers, model_size, num_heads, hidden_size)
        return model(x[..., None])
    van = hk.transform(forward_fn)

    x = jnp.array( np.random.choice(M, size=n, replace=False), dtype=jnp.float64 )
    params = van.init(key, x)

    output = van.apply(params, None, x)
    print("x:", x.astype(int))
    print("x.shape:", x.shape, "output.shape:", output.shape)
    assert output.shape == (n, M)

    random_vec = jnp.array( np.random.randn(n, M) )
    jac = jax.jacrev(lambda x: (van.apply(params, None, x) * random_vec).sum(axis=-1))(x)
    print("jac:", jac)
    print("jac.shape:", jac.shape)
    assert jac.shape == (n, n)

    depends = (jac != 0.).astype(int)
    print(depends)
    assert jnp.allclose(depends, jnp.tril(jnp.ones((n, n)), k=-1))
