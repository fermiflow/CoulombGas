"""
    Autoregressive model of momenta occupation with fixed particle number.

    The present implementation is based on a Transformer composed of causal self-attention layers.

    Adapted from https://github.com/deepmind/dm-haiku/blob/main/examples/transformer/model.py
"""

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import haiku as hk

from typing import Optional

class CausalSelfAttention(hk.MultiHeadAttention):
    """Self attention with a causal mask applied."""
    
    def __call__(self,
                 query: jnp.ndarray,
                 key: Optional[jnp.ndarray] = None,
                 value: Optional[jnp.ndarray] = None,
                ) -> jnp.ndarray:

        key = key if key is not None else query
        value = value if value is not None else query

        seq_len = query.shape[0]
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, ...]

        return super().__call__(query, key, value, mask)

class DenseBlock(hk.Module):
    """A 2-layer MLP which widens then narrows the input."""

    def __init__(self,
                 hidden_size: int,
                 init_scale: float,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.init_scale = init_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        size = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self.init_scale)
        x = hk.Linear(self.hidden_size, w_init=initializer)(x)
        x = jnp.tanh(x)
        return hk.Linear(size, w_init=initializer)(x)

class Transformer(hk.Module):

    def __init__(self,
                 output_size: int,
                 num_layers: int, model_size: int, num_heads: int,
                 hidden_size: int,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.output_size = output_size

        self.num_layers, self.model_size, self.num_heads = \
                num_layers, model_size, num_heads
        if model_size % num_heads != 0:
            raise ValueError("Model_size of the transformer must be divisible "
                    "by the number of heads. Got model_size=%d and num_heads=%d."
                    % (model_size, num_heads))
        self.key_size = model_size // num_heads

        self.hidden_size = hidden_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        init_scale = 0.02 / self.num_layers

        x = hk.Linear(self.model_size,
                      w_init=hk.initializers.VarianceScaling(init_scale, "fan_out"),
                      name="embedding_mlp")(x)
        x = jnp.tanh(x)

        for i in range(self.num_layers):
            x_attn = CausalSelfAttention(self.num_heads,
                                         self.key_size,
                                         init_scale,
                                         name=f"layer{i}_attn")(x)
            x = x + x_attn
            x_dense = DenseBlock(self.hidden_size, init_scale, name=f"layer{i}_mlp")(x)
            x = x + x_dense

        x = jnp.tanh(x)
        x = hk.Linear(self.output_size,
                      w_init=hk.initializers.VarianceScaling(init_scale),
                      name="output_mlp")(x)

        x1init = hk.initializers.TruncatedNormal(jnp.sqrt(init_scale / self.output_size))
        x1hat = hk.get_parameter("x1hat", shape=(self.output_size,), init=x1init)
        x = jnp.vstack((x1hat, x[:-1]))

        return x
