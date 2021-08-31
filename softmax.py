import jax
from jax.config import config
config.update("jax_enable_x64", True)

"""
    Softmax sampler for the state occupation probability.

Parameters:

    logits: unnomalized log probabilities of the state occupation, with shape
        (n_manybody_states,). See function `manybody_orbitals` in orbitals.py
        for the detailed context.
"""

def sampler(logits, key, batch):
    return jax.random.categorical(key, logits, shape=(batch,))

def log_prob(logits, sample):
    logp_full = logits - jax.scipy.special.logsumexp(logits)
    return logp_full[sample]

"""
    Classical score function: logits, sample -> score
    This function can be useful for stochastic reconfiguration, the second-order
optimization algorithm based on classical (as well as quantum) Fisher information matrix.

INPUT:
    logits: (n_manybody_states,)    sample: (batch,)
OUTPUT:
    score: (batch, n_manybody_states)
"""
classical_score_fn = jax.vmap(jax.grad(log_prob), in_axes=(None, 0), out_axes=0)
