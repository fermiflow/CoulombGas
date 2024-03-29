import jax
import jax.numpy as jnp

def make_autoregressive_sampler(network, sp_indices, n, num_states, mask_fn=False):

    def _mask(state_idx):
        mask = jnp.tril(jnp.ones((n, num_states)), k=num_states-n)
        idx_lb = jnp.concatenate( (jnp.array([-1]), state_idx[:-1]) )
        mask = jnp.where(jnp.arange(num_states) > idx_lb[:, None], mask, 0.)
        return mask

    def _logits(params, state_idx):
        """
            Given occupation state indices `state_idx` of the electrons, compute
        the masked logits for the family of autoregressive conditional probabilities.

        Relevant dimensions: (before vmapped)

        INPUT:
            state_idx: (n,), with elements being integers in [0, num_states).
        OUTPUT:
            masked_logits: (n, num_states)
        """

        logits = network.apply(params, None, sp_indices[state_idx])
        mask = _mask(state_idx)
        masked_logits = jnp.where(mask, logits, -1e50)
        return masked_logits

    def sampler(params, key, batch):
        state_indices = jnp.zeros((batch, n), dtype=jnp.int32)
        for i in range(n):
            key, subkey = jax.random.split(key)
            # logits.shape: (batch, n, num_states)
            logits = jax.vmap(_logits, (None, 0), 0)(params, state_indices)
            state_indices = state_indices.at[:, i].set(
                            jax.random.categorical(subkey, logits[:, i, :], axis=-1))
        return state_indices

    def log_prob(params, state_idx):
        logits = _logits(params, state_idx)
        logp = jax.nn.log_softmax(logits, axis=-1)
        logp = logp[jnp.arange(n), state_idx].sum()
        return logp

    if mask_fn:
        # Return the function `_mask` only for test and illustration purpose.
        return _mask, sampler, log_prob
    else:
        return sampler, log_prob


"""
    Classical score function: params, sample -> score
    This function can be useful for stochastic reconfiguration, the second-order
optimization algorithm based on classical Fisher information matrix.

Relevant dimension: (after vmapped)

INPUT:
    params: a pytree    sample: (batch, n), with elements being integers in [0, num_states).
OUTPUT:
    a pytree of the same structure as `params`, in which each leaf node has
an additional leading batch dimension.
"""
make_classical_score = lambda log_prob: jax.vmap(jax.grad(log_prob), (None, 0), 0)


if __name__ == "__main__":
    from jax.config import config
    config.update("jax_enable_x64", True)

    n, num_states = 4, 10
    mask_fn, _, _ = make_autoregressive_sampler(None, None, n, num_states, mask_fn=True)

    state_idx = jnp.array([1, 4, 5, 7])
    mask = mask_fn(state_idx)
    """
        For this example, the resulting mask is illustrated as follows:

                possible state indices
        0   1   2   3   4   5   6   7   8   9
        -------------------------------------
     1  *   *   *   *   *   *   *   0   0   0   1hat
     2  0   0   *   *   *   *   *   *   0   0   2hat(1)
     3  0   0   0   0   0   *   *   *   *   0   3hat(1, 2)
     4  0   0   0   0   0   0   *   *   *   *   4hat(1, 2, 3)

        The symbols "*" and "0" stand for allowed and prohibited states, respectively.
    """
    print("mask:\n", mask)
