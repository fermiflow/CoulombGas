import jax
import jax.numpy as jnp

from functools import partial

from MCMC import mcmc

@partial(jax.pmap, axis_name="p",
                   in_axes=(0, None, 0, None, 0, 0, None, None, None),
                   static_broadcasted_argnums=(1, 3),
                   donate_argnums=4)
def sample_stateindices_and_x(key,
                              sampler, params_van,
                              logp, x, params_flow,
                              mc_steps, mc_stddev, L):
    """
        Generate new state_indices of shape (batch, n), as well as coordinate sample
    of shape (batch, n, dim), from the sample of last optimization step.
    """
    key, key_state, key_MCMC = jax.random.split(key, 3)
    batch = x.shape[0]
    state_indices = sampler(params_van, key_state, batch)
    x, accept_rate = mcmc(lambda x: logp(x, params_flow, state_indices), x, key_MCMC, mc_steps, mc_stddev)
    x -= L * jnp.floor(x/L)
    return key, state_indices, x, accept_rate

####################################################################################

from potential import potential_energy

def make_loss(log_prob, logpsi, logpsi_grad_laplacian, kappa, G, L, rs, Vconst, beta):

    def observable_and_lossfn(params_van, params_flow, state_indices, x, key):
        logp_states = log_prob(params_van, state_indices)
        grad, laplacian = logpsi_grad_laplacian(x, params_flow, state_indices, key)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)

        kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))
        potential = potential_energy(x, kappa, G, L, rs) + Vconst
        Eloc = kinetic + potential
        Floc = logp_states / beta + Eloc.real

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
        observable = {"K_mean": K_mean, "K2_mean": K2_mean,
                      "V_mean": V_mean, "V2_mean": V2_mean,
                      "E_mean": E_mean, "E2_mean": E2_mean,
                      "F_mean": F_mean, "F2_mean": F2_mean,
                      "S_mean": S_mean, "S2_mean": S2_mean}

        def classical_lossfn(params_van):
            logp_states = log_prob(params_van, state_indices)

            tv = jax.lax.pmean(jnp.abs(Floc - F_mean).mean(), axis_name="p")
            Floc_clipped = jnp.clip(Floc, F_mean - 5.0*tv, F_mean + 5.0*tv)
            gradF_phi = (logp_states * (Floc_clipped - F_mean)).mean()
            return gradF_phi

        def quantum_lossfn(params_flow):
            logpsix = logpsi(x, params_flow, state_indices)

            tv = jax.lax.pmean(jnp.abs(Eloc - E_mean).mean(), axis_name="p")
            Eloc_clipped = jnp.clip(Eloc, E_mean - 5.0*tv, E_mean + 5.0*tv)
            gradF_theta = 2 * (logpsix * (Eloc_clipped - E_mean).conj()).real.mean()
            return gradF_theta

        return observable, classical_lossfn, quantum_lossfn

    return observable_and_lossfn
