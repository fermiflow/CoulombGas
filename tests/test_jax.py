import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
import time

key = jax.random.PRNGKey(10)

def test_jit1():
    def f(x):
        x -= x.mean(0)
        x /= x.std(0)
        return x
    f_jit = jax.jit(f)

    n = 100
    x = jax.random.normal(key, (100, 5))
    print("x:", x.shape, x.device(), x.dtype)

    print("--- Original jax version ---")
    for i in range(n):
        start = time.time()
        y = f(x)
        print(i+1, "time to take (ms):", 1000 * (time.time() - start))

    print("--- With jit ---")
    for i in range(n):
        start = time.time()
        y_jit = f_jit(x)
        print(i+1, "time to take (ms):", 1000 * (time.time() - start))

    assert jnp.allclose(y, y_jit)

def test_jit2():
    @jax.jit
    def f(x):
        print("x:", x)
        x -= x.mean(0)
        x /= x.std(0)
        return x

    n = 100
    x1 = jax.random.normal(key, (100, 5))
    print("x1:", x1.shape, x1.device())
    print("--- function f applied to x1 ---")
    for i in range(n):
        start = time.time()
        f(x1)
        print(i+1, "time to take (ms):", 1000 * (time.time() - start))

    x2 = jax.random.normal(key, (200, 30))
    print("x2:", x2.shape, x2.device())
    print("--- function f applied to x2 ---")
    for i in range(n):
        start = time.time()
        f(x2)
        print(i+1, "time to take (ms):", 1000 * (time.time() - start))

def test_jit3():
    @jax.jit
    def f(x):
        return x.reshape( (np.prod(x.shape),) )

    n = 100
    x1 = jax.random.normal(key, (20, 30))
    print("x1:", x1.shape, x1.device())
    print("--- function f applied to x1 ---")
    for i in range(n):
        start = time.time()
        f(x1)
        print(i+1, "time to take (ms):", 1000 * (time.time() - start))

    x2 = jax.random.normal(key, (40, 50))
    print("x2:", x2.shape, x2.device())
    print("--- function f applied to x2 ---")
    for i in range(n):
        start = time.time()
        f(x2)
        print(i+1, "time to take (ms):", 1000 * (time.time() - start))

def test_jit_foriloop():
    @jax.jit
    def sum1(n):
        val = 0
        for i in range(n): val += i
        return val
    @jax.jit
    def sum2(n):
        body_fun = lambda i, val: val + i
        return jax.lax.fori_loop(0, n, body_fun, 0)

    n = 10
    try:
        print("sum1(%d):" % n, sum1(n))     # sum1(n) will fail!
    except Exception as e:
        print("Exception message:\n{}".format(e))
    print("sum2(%d):" % n, sum2(n))     # sum2(n) using jax.lax.fori_loop is OK!!!

####################################################################################

def test_stop_gradient():
    fun1 = lambda x: (jnp.sin(x) * x**3).sum()
    fun2 = lambda x: (jnp.sin(x) * jax.lax.stop_gradient(x**3)).sum()
    x = jnp.array( np.random.randn(3, 4) )

    grad1 = jax.grad(fun1)(x)
    grad1_analytic = jnp.cos(x) * x**3 + jnp.sin(x) * 3*x**2
    assert jnp.allclose(grad1, grad1_analytic)

    grad2 = jax.grad(fun2)(x)
    grad2_analytic = jnp.cos(x) * x**3
    assert jnp.allclose(grad2, grad2_analytic)

####################################################################################

def hvp(f, primals, tangents):
    """ Hessian-vector product: forward-over-reverse """
    _, tangent_out = jax.jvp(jax.grad(f), primals, tangents)
    return tangent_out

def test_hvp():
    f = lambda x: x[0]**3 + jnp.sin(x[0]) * x[1]**2
    x = jnp.array( np.random.randn(2) )
    v = jnp.array( np.random.randn(2) )

    hessianf_v_p = hvp(f, (x,), (v,))

    hessianf = jax.hessian(f)(x)
    hessianf_analytic = jnp.array([[6*x[0]-jnp.sin(x[0])*x[1]**2, 2*x[1]*jnp.cos(x[0])],
                                   [2*x[1]*jnp.cos(x[0]), 2*jnp.sin(x[0])]])
    assert jnp.allclose(hessianf, hessianf_analytic)

    assert jnp.allclose(hessianf_v_p, hessianf.dot(v))

####################################################################################
# Test various implementations of divergence of a R^n -> R^n function.

def div_jvp_fori(f):
    def div_f(x):
        n, = x.shape
        eye = jnp.eye(n)

        def body_fun(i, val):
            primal, tangent = jax.jvp(f, (x,), (eye[i],))
            return val + tangent[i]

        return jax.lax.fori_loop(0, n, body_fun, 0.)
    return div_f

def div_jvp_vmap(f):
    def div_f(x):
        def body_fun(x, basevec):
            _, tangent = jax.jvp(f, (x,), (basevec,))
            return (tangent * basevec).sum()
        return jax.vmap(body_fun, (None, 0), 0)(x, jnp.eye(x.shape[0])).sum()
    return div_f

def test_div():
    f = lambda x: x**3 * jnp.sin(x)
    x = jnp.array( np.random.randn(1000) )
    divf_analytic = (3*x**2*jnp.sin(x) + x**3*jnp.cos(x)).sum()
    divf_jvp_fori = div_jvp_fori(f)
    divf_jvp_fori_jit = jax.jit(divf_jvp_fori)
    divf_jvp_vmap = div_jvp_vmap(f)
    assert jnp.allclose(divf_jvp_fori(x), divf_analytic)
    assert jnp.allclose(divf_jvp_fori_jit(x), divf_analytic)
    assert jnp.allclose(divf_jvp_vmap(x), divf_analytic)

    for _ in range(50):
        start = time.time()
        divf_jvp_fori(x)
        print(time.time() - start, end="\t\t")

        start = time.time()
        divf_jvp_fori_jit(x)
        print(time.time() - start, end="\t\t")

        start = time.time()
        divf_jvp_vmap(x)
        print(time.time() - start)
