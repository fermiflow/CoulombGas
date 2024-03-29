import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ..sampler import make_autoregressive_sampler

def make_loss(log_prob, Es, beta):
    
    def loss_fn(params, state_indices):
        logp = log_prob(params, state_indices)
        E = Es[state_indices].sum(axis=-1)
        F = jax.lax.stop_gradient(logp / beta + E)

        E_mean = E.mean()
        F_mean = F.mean()
        S_mean = -logp.mean()
        E_std = E.std()
        F_std = F.std()
        S_std = (-logp).std()

        gradF = (logp * (F - F_mean)).mean()

        auxiliary_data = {"E_mean": E_mean, "E_std": E_std,
                          "F_mean": F_mean, "F_std": F_std,
                          "S_mean": S_mean, "S_std": S_std,
                         }

        return gradF, auxiliary_data

    return loss_fn

def pretrain(van, params_van,
             n, dim, Theta, Emax, twist,
             path, key,
             lr, sr, damping, max_norm,
             batch, epoch=10000):

    # We recompute the relevant system parameters here for convenience.
    if dim == 3:
        L = (4/3*jnp.pi*n)**(1/3)
        beta = 1 / ((4.5*jnp.pi)**(2/3) * Theta)
    elif dim == 2:
        L = jnp.sqrt(jnp.pi*n)
        beta = 1/ (4 * Theta)

    from ..orbitals import sp_orbitals, twist_sort
    sp_indices, _ = sp_orbitals(dim, Emax)
    sp_indices_twist, Es_twist = twist_sort(sp_indices, twist)
    del sp_indices
    sp_indices_twist = jnp.array(sp_indices_twist)[::-1]
    Es_twist = (2*jnp.pi/L)**2 * jnp.array(Es_twist)[::-1]

    from mpmath import mpf, mp
    from .analytic import Z_E
    F, E, S = Z_E(n, dim, mpf(str(Theta)), [mpf(twist_i) for twist_i in np.array(twist)], Emax)
    print("Analytic results for the thermodynamic quantities: "
            "F: %s, E: %s, S: %s" % (mp.nstr(F), mp.nstr(E), mp.nstr(S)))

    num_states = Es_twist.size
    sampler, log_prob_novmap = make_autoregressive_sampler(van, sp_indices_twist, n, num_states)
    log_prob = jax.vmap(log_prob_novmap, (None, 0), 0)

    loss_fn = make_loss(log_prob, Es_twist, beta)

    import optax
    if sr:
        from ..sampler import make_classical_score
        score_fn = make_classical_score(log_prob_novmap)
        from ..sr import fisher_sr
        optimizer = fisher_sr(score_fn, damping, max_norm)
        print("Optimizer fisher_sr: damping = %.5f, max_norm = %.5f." % (damping, max_norm))
    else:
        optimizer = optax.adam(lr)
        print("Optimizer adam: lr = %.3f." % lr)
    opt_state = optimizer.init(params_van)

    @jax.jit
    def update(params_van, opt_state, key):
        key, subkey = jax.random.split(key)
        state_indices = sampler(params_van, subkey, batch)

        grads, aux = jax.grad(loss_fn, argnums=0, has_aux=True)(params_van, state_indices)
        updates, opt_state = optimizer.update(grads, opt_state,
                                params=(params_van, state_indices) if sr else None)
        params_van = optax.apply_updates(params_van, updates)

        return params_van, opt_state, key, aux

    import os
    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w", buffering=1, newline="\n")

    for i in range(1, epoch+1):
        params_van, opt_state, key, aux = update(params_van, opt_state, key)
        E, E_std, F, F_std, S, S_std = aux["E_mean"], aux["E_std"], \
                                       aux["F_mean"], aux["F_std"], \
                                       aux["S_mean"], aux["S_std"]
        print("iter: %04d" % i,
                "F:", F, "F_std:", F_std / jnp.sqrt(batch),
                "E:", E, "E_std:", E_std / jnp.sqrt(batch),
                "S:", S, "S_std:", S_std / jnp.sqrt(batch))

        f.write( ("%6d" + "  %.6f"*6 + "\n") % (i, F, F_std / jnp.sqrt(batch),
                                                   E, E_std / jnp.sqrt(batch),
                                                   S, S_std / jnp.sqrt(batch)) )

    return params_van