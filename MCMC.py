import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

from functools import partial

@partial(jax.jit, static_argnums=0)
def mcmc(logp_fn, x_init, key, mc_steps, mc_stddev=0.02):
    """
        Markov Chain Monte Carlo sampling algorithm.

    INPUT:
        logp_fn: callable that evaluate log-probability of a batch of configuration x.
            The signature is logp_fn(x), where x has shape (batch, n, dim).
        x_init: initial value of x, with shape (batch, n, dim).
        key: initial PRNG key.
        mc_steps: total number of Monte Carlo steps.
        mc_stddev: standard deviation of the Gaussian proposal.

    OUTPUT:
        x: resulting batch samples, with the same shape as `x_init`.
    """
    def step(i, state):
        x, logp, key = state
        key, key_proposal, key_accept = jax.random.split(key, 3)

        x_proposal = x + mc_stddev * jax.random.normal(key_proposal, x.shape)
        logp_proposal = logp_fn(x_proposal)
        ratio = jnp.exp(logp_proposal - logp)
        accept = jax.random.uniform(key_accept, ratio.shape) < ratio

        x_new = jnp.where(accept[:, None, None], x_proposal, x)
        logp_new = jnp.where(accept, logp_proposal, logp)
        return x_new, logp_new, key
    
    logp_init = logp_fn(x_init)
    x, logp, key = lax.fori_loop(0, mc_steps, step, (x_init, logp_init, key))
    return x
