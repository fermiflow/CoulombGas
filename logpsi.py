import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from slater import logslaterdet
from functools import partial

def make_logpsi(flow, sp_indices, L):

    def logpsi(x, params, state_idx):

        """
            Generic function that computes ln Psi_n(x) given a single-particle orbital
        `state_idx`, a set of electron coordinates `x`, and flow parameters `params`.

        INPUT:
            x: (n, dim)     state_idx: (n,), with elements being integers in [0, num_states).

        OUTPUT:
            a single complex number ln Psi_n(x), given in the form of a 2-tuple (real, imag).
        """

        z = flow.apply(params, None, x)
        log_phi = logslaterdet(sp_indices[state_idx], z, L)

        n, dim = x.shape
        x_flatten = x.reshape(-1)
        flow_flatten = lambda x: flow.apply(params, None, x.reshape(n, dim)).reshape(-1)
        jac = jax.jacfwd(flow_flatten)(x_flatten)
        _, logjacdet = jnp.linalg.slogdet(jac)
        return jnp.stack([log_phi.real + 0.5*logjacdet,
                          log_phi.imag])

    return logpsi

def make_logpsi_grad_laplacian(logpsi, forloop=True, key=None):

    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=0)
    def logpsi_vmapped(x, params, state_idx):
        logpsix = logpsi(x, params, state_idx)
        return logpsix[0] + 1j * logpsix[1]

    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=0)
    def logpsi_grad_laplacian(x, params, state_idx):
        """
            Computes the gradient and laplacian of logpsi w.r.t. electron coordinates x.
        The final result is in complex form.

        Relevant dimensions: (after vmapped)

        INPUT:
            x: (batch, n, dim)  state_idx: (batch, n)
        OUTPUT:
            grad: (batch, n, dim)   laplacian: (batch,)
        """

        grad = jax.jacrev(logpsi)(x, params, state_idx)
        grad = grad[0] + 1j * grad[1]
        print("Computed gradient.")

        n, dim = x.shape
        x_flatten = x.reshape(-1)
        grad_logpsi = jax.jacrev(lambda x: logpsi(x.reshape(n, dim), params, state_idx))

        def _laplacian(x):
            if forloop:
                print("forloop version...")
                def body_fun(i, val):
                    _, tangent = jax.jvp(grad_logpsi, (x,), (eye[i],))
                    return val + tangent[0, i] + 1j * tangent[1, i]
                eye = jnp.eye(x.shape[0])
                laplacian = jax.lax.fori_loop(0, x.shape[0], body_fun, 0.+0.j)
            else:
                print("vmap version...")
                def body_fun(x, basevec):
                    _, tangent = jax.jvp(grad_logpsi, (x,), (basevec,))
                    return (tangent * basevec).sum(axis=-1)
                eye = jnp.eye(x.shape[0])
                laplacian = jax.vmap(body_fun, (None, 1), 1)(x, eye).sum(axis=-1)
                laplacian = laplacian[0] + 1j * laplacian[1]
            return laplacian

        laplacian = _laplacian(x_flatten)
        print("Computed laplacian.")

        return grad, laplacian

    def logpsi_grad_laplacian_hutchinson(x, params, state_indices):

        v = jax.random.normal(key, x.shape)

        @partial(jax.vmap, in_axes=(0, None, 0, 0), out_axes=0)
        def logpsi_grad_random_laplacian(x, params, state_idx, v):
            """
                Compute the laplacian as a random variable `v^T hessian(ln Psi_n(x)) v`
            using the Hutchinson's trick.

                The argument `v` is a random "vector" that has the same shape as `x`,
            i.e., (after vmapped) (batch, n, dim).
            """

            grad, hvp = jax.jvp( jax.jacrev(lambda x: logpsi(x, params, state_idx)),
                                 (x,), (v,) )

            grad = grad[0] + 1j * grad[1]
            print("Computed gradient.")

            random_laplacian = (hvp * v).sum(axis=(-2, -1))
            random_laplacian = random_laplacian[0] + 1j * random_laplacian[1]
            print("Computed Hutchinson's estimator of laplacian.")

            return grad, random_laplacian

        return logpsi_grad_random_laplacian(x, params, state_indices, v)

    return logpsi_vmapped, \
           (logpsi_grad_laplacian_hutchinson if key is not None else logpsi_grad_laplacian)

def make_logp(logpsi):

    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=0)
    def logp(x, params, state_idx):
        """ logp = logpsi + logpsi* = 2 Re logpsi """
        return 2 * logpsi(x, params, state_idx)[0]

    return logp

def make_quantum_score(logpsi):

    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=0)
    def quantum_score_fn(x, params, state_idx):
        """
            Computes the "quantum score function", i.e., the gradient of ln Psi_n(x)
        w.r.t. the flow parameters.
            This function can be useful for stochastic reconfiguraton, the
        second-order optimization algorithm based on quantum (as well as classical)
        Fisher information matrix.

        Relevant dimension: (after vmapped)

        OUTPUT:
            a pytree of the same structure as `params`, in which each leaf node has
        an additional leading batch dimension.
        """
        grad_params = jax.jacrev(logpsi, argnums=1)(x, params, state_idx)
        return jax.tree_map(lambda jac: jac[0] + 1j * jac[1], grad_params)

    return quantum_score_fn
