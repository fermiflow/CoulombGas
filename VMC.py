import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from functools import partial

from softmax import sampler
from MCMC import mcmc

#@partial(jax.pmap, in_axes=(0, 0, 0, 0, None, None), static_broadcasted_argnums=4)
def sample_stateindices_and_x(key, logits,
                              logp, x, params, mc_steps, mc_stddev, L):
    """
        Generate new state_indices of shape (batch,), as well as coordinate sample
    of shape (batch, n, dim), from the sample of last optimization step.
    """
    key, key_state, key_MCMC = jax.random.split(key, 3)
    batch = x.shape[0]
    state_indices = sampler(logits, key_state, batch)
    x = mcmc(lambda x: logp(x, params, state_indices), x, key_MCMC, mc_steps, mc_stddev)
    x -= L * jnp.floor(x/L)
    return key, state_indices, x

####################################################################################

from softmax import log_prob
from potential import potential_energy

def make_loss(logpsi_grad_laplacian, kappa, G, L, rs, Vconst, beta):

    def loss_fn(logits, params, state_indices, x):
        logp_states = log_prob(logits, state_indices)

        logpsi, grad, laplacian = logpsi_grad_laplacian(x, params, state_indices)
        print("logpsi.shape:", logpsi.shape)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)

        kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))
        potential = potential_energy(x, kappa, G, L, rs) + Vconst
        Eloc = jax.lax.stop_gradient(kinetic + potential)
        Floc = jax.lax.stop_gradient(logp_states / beta + Eloc.real)

        K_mean, K2_mean, V_mean, V2_mean, \
        E_mean, E2_mean, F_mean, F2_mean, S_mean, S2_mean = \
        jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
                     (kinetic.real.mean(), (kinetic.real**2).mean(),
                      potential.mean(), (potential**2).mean(),
                      Eloc.real.mean(), (Eloc.real**2).mean(),
                      Floc.mean(), (Floc**2).mean(),
                      -logp_states.mean(), (logp_states**2).mean()
                     )
                    )
        logp_states_all = log_prob(logits, jnp.arange(logits.shape[0]))
        S_logits = - (jnp.exp(logp_states_all) * logp_states_all).sum()

        tv = jax.lax.pmean(jnp.abs(Floc - F_mean).mean(), axis_name="p")
        Floc_clipped = jnp.clip(Floc, F_mean - 5.0*tv, F_mean + 5.0*tv)
        gradF_phi = (logp_states * (Floc_clipped - F_mean)).mean()

        tv = jax.lax.pmean(jnp.abs(Eloc - E_mean).mean(), axis_name="p")
        Eloc_clipped = jnp.clip(Eloc, E_mean - 5.0*tv, E_mean + 5.0*tv)
        gradF_theta = 2 * (logpsi * (Eloc_clipped - E_mean).conj()).real.mean()

        auxiliary_data = {"statistics":
                            {"K_mean": K_mean, "K2_mean": K2_mean,
                             "V_mean": V_mean, "V2_mean": V2_mean,
                             "E_mean": E_mean, "E2_mean": E2_mean,
                             "F_mean": F_mean, "F2_mean": F2_mean,
                             "S_mean": S_mean, "S2_mean": S2_mean,
                             "S_logits": S_logits},
                          "logpsi": logpsi,
                          "Eloc_real": Eloc.real,
                         }

        return gradF_phi + gradF_theta, auxiliary_data

    return loss_fn
