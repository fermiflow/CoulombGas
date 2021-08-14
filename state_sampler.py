import jax
from jax.config import config
config.update("jax_enable_x64", True)

def make_softmax_sampler(logits):
    """
        Softmax sampler for the state occupation probability.

        logits: unnomalized log probabilities of the state occupation, with shape
            (n_manybody_states,). See function `manybody_orbitals` in orbitals.py
            for the detailed context.
    """
    def sampler(key, batch):
        return jax.random.categorical(key, logits, shape=(batch,))

    def log_prob(sample):
        logp_full = logits - jax.scipy.special.logsumexp(logits)
        return logp_full[sample]

    return sampler, log_prob
