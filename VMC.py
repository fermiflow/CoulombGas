import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from slater import logslaterdet
from functools import partial

def make_logpsi_logp(flow, manybody_indices, L):
    def logpsi(x, params, state_idx):
        """
            x: (n, dim)
            params: flow parameters.
            state_idx: single integer index specifying an n-electron many-body
                slater determinant.
        """
        z = flow.apply(params, None, x)
        log_phi = logslaterdet(manybody_indices[state_idx], z, L)

        n, dim = x.shape
        x_flatten = x.reshape(-1)
        flow_flatten = lambda x: flow.apply(params, None, x.reshape(n, dim)).reshape(-1)
        jac = jax.jacfwd(flow_flatten)(x_flatten)
        _, logjacdet = jnp.linalg.slogdet(jac)
        return jnp.stack([log_phi.real + 0.5*logjacdet,
                          log_phi.imag])

    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=0)
    def logp(x, params, state_idx):
        """ logp = logpsi + logpsi* = 2 Re logpsi """
        return 2 * logpsi(x, params, state_idx)[0]

    return logpsi, logp

@partial(jax.vmap, in_axes=(0, None, 0, None), out_axes=0)
def logpsi_grad_laplacian(x, params, state_idx, logpsi):
    """
        (After vmapped) Compute logpsi and its gradient and laplacian of a batched
    coordinate sample `x` with shape (batch, n, dim).

    OUTPUT:
        logpsix: (batch,)
        grad: (batch, n, dim)
        laplacian: (batch,)
    """
    logpsix = logpsi(x, params, state_idx)
    logpsix = logpsix[0] + 1j * logpsix[1]
    print("Computed logpsi.")

    grad = jax.jacrev(logpsi)(x, params, state_idx)
    grad = grad[0] + 1j * grad[1]
    print("Computed gradient.")

    n, dim = x.shape
    x_flatten = x.reshape(-1)
    grad_logpsi = jax.jacrev(lambda x: logpsi(x.reshape(n, dim), params, state_idx))

    def _laplacian(x):
        def body_fun(x, basevec):
            _, tangent = jax.jvp(grad_logpsi, (x,), (basevec,))
            return (tangent * basevec).sum(axis=-1)
        eye = jnp.eye(x.shape[0])
        laplacian = jax.vmap(body_fun, (None, 1), 1)(x, eye).sum(axis=-1)
        return laplacian

    laplacian = _laplacian(x_flatten)
    laplacian = laplacian[0] + 1j * laplacian[1]
    print("Computed laplacian.")

    return logpsix, grad, laplacian

from softmax import sampler, log_prob

#@partial(jax.pmap, in_axes=(0, 0, 0, 0, None, None), static_broadcasted_argnums=4)
def sample_x(key, x, logits, params, logp, mc_steps):
    """
        Generate new state_indices as well as coordinate sample of shape (batch, n, dim)
    from the sample of last optimization step.
    """
    key, key_state, key_MCMC = jax.random.split(key, 3)
    batch = x.shape[0]
    state_indices = sampler(logits, key_state, batch)
    x = mcmc(lambda x: logp(x, params, state_indices), x, key_MCMC, mc_steps)
    return key, state_indices, x

from potential import potential_energy
from MCMC import mcmc

def make_loss(logp, mc_steps, logpsi,
              kappa, G, Vconst, L, rs, beta
             ):

    def loss_fn(logits, params, key, x):
        key, state_indices, x = sample_x(key, x, logits, params, logp, mc_steps)
        print("Sampled state indices and electron coordinates.")
        state_indices = jax.lax.stop_gradient(state_indices)
        x = jax.lax.stop_gradient(x)

        logp_states = log_prob(logits, state_indices)

        logpsix, grad, laplacian = logpsi_grad_laplacian(x, params, state_indices, logpsi)
        print("logpsix.shape:", logpsix.shape)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)
        kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))
        potential = potential_energy(x, kappa, G, L, rs) + Vconst

        Eloc = jax.lax.stop_gradient(kinetic + potential)
        E_mean = Eloc.real.mean()
        E2_mean = (Eloc.real**2).mean()

        S_mean = -logp_states.mean()
        S2_mean = (logp_states**2).mean()
        logp_states_all = log_prob(logits, jnp.arange(logits.shape[0]))
        S_logits = - (jnp.exp(logp_states_all) * logp_states_all).sum()

        Floc = jax.lax.stop_gradient(logp_states / beta + Eloc.real)
        F_mean = Floc.mean()
        F2_mean = (Floc**2).mean()

        gradF_phi = (logp_states * (Floc - F_mean)).mean()
        gradF_theta = 2 * (logpsix * (Eloc - E_mean).conj()).real.mean()

        auxiliary_data = {"statistics":
                            {"E_mean": E_mean, "E2_mean": E2_mean,
                             "F_mean": F_mean, "F2_mean": F2_mean,
                             "S_mean": S_mean, "S2_mean": S2_mean},
                          "S_logits": S_logits,
                          "key": key,
                          "x": x,
                         }

        return gradF_phi + gradF_theta, auxiliary_data

    return loss_fn
