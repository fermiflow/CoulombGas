import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from optax._src import base

HybridFisherSRState = base.EmptyState

def hybrid_fisher_sr(classical_score_fn, quantum_score_fn, damping, max_norm):
    """
        A second-order stochastic reconfiguration algorithm for the hybrid optimization
    of both a classical probability distribution and a quantum basis wavefunction
    ansatz.

        The design of API signatures is in parallel with the package `optax`.
    """

    def init_fn(params):
        return HybridFisherSRState()

    def update_fn(grads, state, params):
        """
            NOTE: as the computation of (classical and quantum) Fisher information
        metrics calls for the Monte-Carlo sample `state_indices` and `x`, we manually
        place them within the `params` argument.
        """
        grad_logits, grad_params = grads
        logits, params, state_indices, x = params

        grad_logits_raveled, logits_unravel_fn = ravel_pytree(grad_logits)
        grad_params_raveled, params_unravel_fn = ravel_pytree(grad_params)
        print("grad_logits.shape:", grad_logits_raveled.shape)
        print("grad_params.shape:", grad_params_raveled.shape)

        classical_score = classical_score_fn(logits, state_indices)
        quantum_score = quantum_score_fn(x, params, state_indices)
        classical_score_raveled = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(classical_score)
        quantum_score_raveled = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(quantum_score)
        print("classical_score.shape:", classical_score_raveled.shape)
        print("quantum_score.shape:", quantum_score_raveled.shape)

        batch_per_device = classical_score_raveled.shape[0]


        classical_fisher = jax.lax.pmean(
                    classical_score_raveled.T.dot(classical_score_raveled) / batch_per_device,
                    axis_name="p")
        classical_fisher += damping * jnp.eye(classical_fisher.shape[0])
        update_logits_raveled = jax.scipy.linalg.solve(classical_fisher, grad_logits_raveled)
        #scale gradient according to gradnorm
        gnorm = jnp.sum(grad_logits_raveled * update_logits_raveled)
        scale = jnp.minimum(jnp.sqrt(max_norm/gnorm), 1)
        update_logits_raveled *= -scale
        update_logits = logits_unravel_fn(update_logits_raveled)


        quantum_score_raveled -= jax.lax.pmean(quantum_score_raveled.mean(axis=0), axis_name="p")
        quantum_fisher = jax.lax.pmean(
                    quantum_score_raveled.conj().T.dot(quantum_score_raveled).real / batch_per_device,
                    axis_name="p")
        quantum_fisher += damping * jnp.eye(quantum_fisher.shape[0])
        update_params_raveled = jax.scipy.linalg.solve(quantum_fisher, grad_params_raveled)
        #scale gradient according to gradnorm
        gnorm = jnp.sum(grad_params_raveled * update_params_raveled)
        scale = jnp.minimum(jnp.sqrt(max_norm/gnorm), 1)
        update_params_raveled *= -scale
        update_params = params_unravel_fn(update_params_raveled)


        return (update_logits, update_params), state

    return base.GradientTransformation(init_fn, update_fn)
