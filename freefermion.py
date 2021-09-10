import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from sampler import make_autoregressive_sampler

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

def pretrain(van, params_van, Es, beta, n, dim, key, batch, epoch):

    print("Es:", Es, "\nEs.shape:", Es.shape)
    num_states = Es.size
    sampler, log_prob = make_autoregressive_sampler(van, n, num_states)
    loss_fn = make_loss(log_prob, Es, beta)

    import optax
    lr = 3e-3
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params_van)

    @jax.jit
    def update(params_van, opt_state, key):
        key, subkey = jax.random.split(key)
        state_indices = sampler(params_van, subkey, batch)
        print("state_indices:", state_indices, "\nstate_indices.shape:", state_indices.shape)

        grads, aux = jax.grad(loss_fn, argnums=0, has_aux=True)(params_van, state_indices)
        updates, opt_state = optimizer.update(grads, opt_state)
        params_van = optax.apply_updates(params_van, updates)

        return params_van, opt_state, key, aux

    for i in range(1, epoch+1):
        params_van, opt_state, key, aux = update(params_van, opt_state, key)
        E, E_std, F, F_std, S, S_std = aux["E_mean"], aux["E_std"], \
                                       aux["F_mean"], aux["F_std"], \
                                       aux["S_mean"], aux["S_std"]
        print("iter: %04d" % i,
                "F:", F, "F_std:", F_std / jnp.sqrt(batch),
                "E:", E, "E_std:", E_std / jnp.sqrt(batch),
                "S:", S, "S_std:", S_std / jnp.sqrt(batch))

    return params_van
