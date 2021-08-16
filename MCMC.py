import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

from functools import partial

@partial(jax.jit, static_argnums=0)
def mcmc(logp_fn, x_init, key, mc_steps, stddev=0.02):
    """
        Markov Chain Monte Carlo sampling algorithm.

    INPUT:
        logp_fn: callable that evaluate log-probability of a batch of configuration x.
            The signature is logp_fn(x), where x has shape (batch, n, dim).
        x_init: initial value of x, with shape (batch, n, dim).
        key: initial PRNG key.
        mc_steps: total number of Monte Carlo steps.
        stddev: standard deviation of the Gaussian proposal.

    OUTPUT:
        x: resulting batch samples, with the same shape as `x_init`.
    """
    def step(i, state):
        x, logp, key = state
        key, key_proposal, key_accept = jax.random.split(key, 3)

        x_proposal = x + stddev * jax.random.normal(key_proposal, x.shape)
        logp_proposal = logp_fn(x_proposal)
        ratio = jnp.exp(logp_proposal - logp)
        accept = jax.random.uniform(key_accept, ratio.shape) < ratio

        x_new = jnp.where(accept[:, None, None], x_proposal, x)
        logp_new = jnp.where(accept, logp_proposal, logp)
        return x_new, logp_new, key
    
    logp_init = logp_fn(x_init)
    x, logp, key = lax.fori_loop(0, mc_steps, step, (x_init, logp_init, key))
    return x

if __name__ == "__main__":
    import numpy as np
    import haiku as hk
    from flow import FermiNet
    from orbitals import manybody_orbitals
    from funs import make_logpsi_logp

    depth = 3
    spsize, tpsize = 16, 16
    L = 1.234

    n, dim = 7, 3
    
    def flow_fn(x):
        model = FermiNet(depth, spsize, tpsize, L)
        return model(x.reshape(n, dim)).reshape(-1)
    flow = hk.transform(flow_fn)
    x = jnp.array( np.random.uniform(0, L, n*dim) )
    key = jax.random.PRNGKey(42)
    params = flow.init(key, x)

    Ecut = 2
    manybody_indices, manybody_Es = manybody_orbitals(n, dim, Ecut)
    print("manybody_indices.shape:", manybody_indices.shape)
    state_idx = 624
    print("manybody_indices[%d]:\n" % state_idx, manybody_indices[state_idx])
    
    logpsi, logp = make_logpsi_logp(flow, manybody_indices, n, dim, L)
    logpsix = logpsi(x, params, state_idx)
    print("logpsix:", logpsix)
    print("logpsix.shape:", logpsix.shape)

    batch = 100
    batch_logp = jax.vmap(logp, (0, None, None), 0)
    x_init = jnp.array( np.random.uniform(0, L, (batch, n*dim)) )
    logpx_init = batch_logp(x_init, params, state_idx)
    print("logpx_init:", logpx_init)
    print("logpx_init.shape:", logpx_init.shape)

    mc_steps = 1000
    x_sample = mcmc(lambda x: batch_logp(x, params, state_idx), x_init, key, mc_steps)
    print("x_sample:", x_sample)
    print("x_sample.shape:", x_sample.shape)
