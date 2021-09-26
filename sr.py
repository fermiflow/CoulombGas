"""
    Second-order optimization algorithm using stochastic reconfiguration.
    The design of API signatures is in parallel with the package `optax`.
"""
import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from optax._src import base

FisherSRState = base.EmptyState

def fisher_sr(score_fn, damping, max_norm):
    """
        SR for a purely classical probabilistic model, which is also known as the
    natural gradient descent in machine-learning literatures.
    """

    def init_fn(params):
        return FisherSRState()

    def update_fn(grads, state, params):
        """
            NOTE: as the computation of Fisher information metric calls for the
        Monte-Carlo sample `state_indices`, we manually place them within the
        `params` argument.
        """
        params, state_indices = params

        grads_raveled, grads_unravel_fn = ravel_pytree(grads)
        print("grads.shape:", grads_raveled.shape)

        score = score_fn(params, state_indices)
        score_raveled = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(score)
        print("score.shape:", score_raveled.shape)

        batch_per_device = score_raveled.shape[0]

        fisher = score_raveled.T.dot(score_raveled) / batch_per_device
        fisher += damping * jnp.eye(fisher.shape[0])
        updates_raveled = jax.scipy.linalg.solve(fisher, grads_raveled)
        #scale gradient according to gradnorm
        gnorm = jnp.sum(grads_raveled * updates_raveled)
        scale = jnp.minimum(jnp.sqrt(max_norm/gnorm), 1)
        updates_raveled *= -scale
        updates = grads_unravel_fn(updates_raveled)

        return updates, state

    return base.GradientTransformation(init_fn, update_fn)

####################################################################################

HybridFisherSRState = base.EmptyState

def hybrid_fisher_sr(classical_score_fn, quantum_score_fn, damping, max_norm):
    """
        Hybrid SR for both a classical probabilistic model and a set of
    quantum basis wavefunction ansatz.
    """

    def init_fn(params):
        return HybridFisherSRState()

    def fishers_fn(params_van, params_flow, state_indices, x):
        classical_score = classical_score_fn(params_van, state_indices)
        classical_score = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(classical_score)

        quantum_score = quantum_score_fn(x, params_flow, state_indices)
        quantum_score = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(quantum_score)
        print("classical_score.shape:", classical_score.shape)
        print("quantum_score.shape:", quantum_score.shape)
        quantum_score_mean = jax.lax.pmean(quantum_score.mean(axis=0), axis_name="p")

        batch_per_device = classical_score.shape[0]

        classical_fisher = jax.lax.pmean(
                    classical_score.T.dot(classical_score) / batch_per_device,
                    axis_name="p")
        quantum_fisher = jax.lax.pmean(
                    quantum_score.conj().T.dot(quantum_score).real / batch_per_device,
                    axis_name="p")

        return classical_fisher, quantum_fisher, quantum_score_mean

    def update_fn(grads, state, params):
        """
            NOTE: as the computation of (classical and quantum) Fisher information
        metrics calls for the Monte-Carlo sample `state_indices` and `x`, we manually
        place them within the `params` argument.
        """
        grad_params_van, grad_params_flow = grads
        classical_fisher, quantum_fisher, quantum_score_mean = params
        quantum_fisher -= (quantum_score_mean.conj()[:, None] * quantum_score_mean).real

        grad_params_van_raveled, params_van_unravel_fn = ravel_pytree(grad_params_van)
        grad_params_flow_raveled, params_flow_unravel_fn = ravel_pytree(grad_params_flow)
        print("grad_params_van.shape:", grad_params_van_raveled.shape)
        print("grad_params_flow.shape:", grad_params_flow_raveled.shape)


        classical_fisher += damping * jnp.eye(classical_fisher.shape[0])
        update_params_van_raveled = jax.scipy.linalg.solve(classical_fisher, grad_params_van_raveled)
        #scale gradient according to gradnorm
        gnorm = jnp.sum(grad_params_van_raveled * update_params_van_raveled)
        scale = jnp.minimum(jnp.sqrt(max_norm/gnorm), 1)
        update_params_van_raveled *= -scale
        update_params_van = params_van_unravel_fn(update_params_van_raveled)


        quantum_fisher += damping * jnp.eye(quantum_fisher.shape[0])
        update_params_flow_raveled = jax.scipy.linalg.solve(quantum_fisher, grad_params_flow_raveled)
        #scale gradient according to gradnorm
        gnorm = jnp.sum(grad_params_flow_raveled * update_params_flow_raveled)
        scale = jnp.minimum(jnp.sqrt(max_norm/gnorm), 1)
        update_params_flow_raveled *= -scale
        update_params_flow = params_flow_unravel_fn(update_params_flow_raveled)


        return (update_params_van, update_params_flow), state

    return fishers_fn, base.GradientTransformation(init_fn, update_fn)
