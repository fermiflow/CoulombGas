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
        x = jax.nn.gelu(x)
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

    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        h = hk.Linear(self.model_size, name="embedding_mlp")(h)
        h = jax.nn.gelu(h)

        init_scale = 2. / self.num_layers

        for i in range(self.num_layers):
            h_attn = CausalSelfAttention(self.num_heads,
                                         self.key_size,
                                         init_scale,
                                         name=f'layer{i}_attn')(h)
            h = h + h_attn
            h_dense = DenseBlock(self.hidden_size, init_scale, name=f'layer{i}_mlp')(h)
            h = h + h_dense

        h = hk.Linear(self.output_size, name="output_mlp")(h)

        return h
