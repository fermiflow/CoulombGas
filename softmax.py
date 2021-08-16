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
