import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from softmax import sampler, log_prob
import numpy as np

def test_softmax():
    N = 10
    logits = jnp.array( np.random.randn(N) )

    p_full = jnp.exp(log_prob(logits, jnp.arange(N)))
    assert jnp.allclose(p_full, jnp.exp(logits) / jnp.exp(logits).sum())
    assert jnp.allclose(p_full.sum(), 1.)

    key = jax.random.PRNGKey(42)
    batch = 100
    sample = sampler(logits, key, batch)

    print("logp_full:", log_prob(logits, jnp.arange(N)))
    print("sample:", sample)
    print("logp:", log_prob(logits, sample))

def test_softmax_AD():
    """
        This test demonstrates that putting the sampler inside a function does not
    affect its AD behavior.
    """
    N = 10
    logits = jnp.array( np.random.randn(N) )
    key = jax.random.PRNGKey(42)
    batch = 100

    # put logits -> sample inside the function.
    def fun1(logits):
        sample = sampler(logits, key, batch)
        return log_prob(logits, sample).sum()
    grad1 = jax.grad(fun1)(logits)
    print("grad1:", grad1)

    # put logits -> sample outside the function.
    sample = sampler(logits, key, batch)
    fun2 = lambda logits: log_prob(logits, sample).sum()
    grad2 = jax.grad(fun2)(logits)
    print("grad2:", grad2)

    # analytic result.
    grad_analytic = jnp.array([ (sample == i).sum() for i in range(N) ]) \
                    - jnp.exp(logits) / jnp.exp(logits).sum() * batch
    print("grad_analytic:", grad_analytic)

    assert jnp.allclose(grad1, grad2)
    assert jnp.allclose(grad1, grad_analytic)
