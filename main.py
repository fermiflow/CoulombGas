import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

key = jax.random.PRNGKey(42)

import argparse
parser = argparse.ArgumentParser(description="Finite-temperature VMC for homogeneous electron gas")

#physical parameters
parser.add_argument("--n", type=int, default=13, help="total number of electrons")
parser.add_argument("--dim", type=int, default=2, help="spatial dimension")
parser.add_argument("--rs", type=float, default=1.0, help="rs")
parser.add_argument("--Theta", type=float, default=0.05, help="dimensionless temperature T/Ef")
parser.add_argument("--Ecut", type=float, default=2.0, help="energy cutoff for the many-body slater determinants")

parser.add_argument("--Gmax", type=int, default=15, help="k-space cutoff in the ewald summation of Coulomb potential")
parser.add_argument("--kappa", type=float, default=10, help="screening parameter (in unit of 1/L) in Ewald summation")

#mcmc parameters
parser.add_argument("--mc_therm", type=int, default=10, help="thermal_steps")
parser.add_argument("--mc_width", type=float, default=0.1, help="mcmc update width")
parser.add_argument("--mc_steps", type=int, default=50, help="mcmc update steps")

#training parameters
parser.add_argument("--batch", type=int, default=1024, help="batch size")
#parser.add_argument("--k_steps", type=int, default=1, help="accumulation steps")
#parser.add_argument("--damping", type=float, default=1e-3, help="damping")
#parser.add_argument("--max_norm", type=float, default=1e-3, help="gradnorm")
#parser.add_argument("--use_sparse_solver", action='store_true',  help="")
parser.add_argument("--epochs", type=int, default=10000, help="epochs")

#model parameters
#parser.add_argument("--steps", type=int, default=2, help="steps")
parser.add_argument("--depth", type=int, default=2, help="depth")
parser.add_argument("--spsize", type=int, default=16, help="spsize")
parser.add_argument("--tpsize", type=int, default=16, help="tpsize")

args = parser.parse_args()

n, dim = args.n, args.dim
if dim == 3:
    L = (4/3*jnp.pi*n)**(1/3)
    beta = 1 / ((4.5*jnp.pi)**(2/3) * args.Theta)
elif dim == 2:
    L = jnp.sqrt(jnp.pi*n)
    beta = 1/ (4 * args.Theta)
print("n = %d, dim = %d, L = %f" % (n, dim, L))

beta = 1.0
print("beta = %f" % beta)

####################################################################################

print("Initialize many-body state distribution...")

from orbitals import manybody_orbitals
manybody_indices, manybody_Es = manybody_orbitals(n, dim, args.Ecut)
manybody_indices, manybody_Es = jnp.array(manybody_indices), jnp.array(manybody_Es)
print("manybody_indices.shape:", manybody_indices.shape)
logits = - beta * manybody_Es * (2*jnp.pi/L)**2
logits -= jax.scipy.special.logsumexp(logits)

####################################################################################

print("Initialize normalizing flow...")

import haiku as hk
from flow import FermiNet
def flow_fn(x):
    model = FermiNet(args.depth, args.spsize, args.tpsize, L)
    return model(x)
flow = hk.transform(flow_fn)
x_dummy = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
params = flow.init(key, x_dummy)

from VMC import make_logpsi_logp
logpsi, logp = make_logpsi_logp(flow, manybody_indices, L)

####################################################################################

print("Initialize relevant quantities for Ewald summation...")

from potential import kpoints, Madelung
G = kpoints(dim, args.Gmax)
Vconst = n * args.rs/L * Madelung(dim, args.kappa, G)
print("(scaled) Vconst:", Vconst/(n*args.rs/L))

####################################################################################

print("Initialize optimizer...")
import optax
from utils import shard, replicate

num_devices = jax.device_count()

lr = 1e-2
optimizer = optax.adam(lr)
opt_state = optimizer.init((logits, params))
opt_state = replicate(opt_state, num_devices)

####################################################################################

print("Generate initial key and coordinate sample...")

print("Number of GPU devices:", num_devices)
if args.batch % num_devices != 0:
    raise ValueError("Batch size must be divisible by the number of GPU devices. "
                     "Got batch = %d for %d devices now." % (args.batch, num_devices))
batch_per_device = args.batch // num_devices

x = jax.random.uniform(key, (num_devices, batch_per_device, n, dim), minval=0., maxval=L)
keys = jax.random.split(key, num_devices)

x, keys = shard(x), shard(keys)
logits, params = replicate((logits, params), num_devices)
print("keys:", keys, "\nshape:", keys.shape, "\t\ttype:", type(keys))
print("x:", x, "\nshape:", x.shape, "\t\ttype:", type(x))

from VMC import sample_x
pmap_sample_x = jax.pmap(sample_x, in_axes=(0, 0, 0, 0, None, None),
                                   static_broadcasted_argnums=4)
for i in range(args.mc_therm):
    print("---- thermal step %d ----" % (i+1))
    keys, _, x = pmap_sample_x(keys, x, logits, params, logp, args.mc_steps)
print("keys:", keys, "\nshape:", keys.shape, "\t\ttype:", type(keys))
print("x:", x, "\nshape:", x.shape, "\t\ttype:", type(x))

####################################################################################

from VMC import make_loss
loss_fn = make_loss(logp, args.mc_steps, logpsi,
          args.kappa, G, Vconst, L, args.rs, beta)

from functools import partial

@partial(jax.pmap, axis_name="p")
def update(logits, params, opt_state, key, x):
    grads, aux = jax.grad(loss_fn, argnums=(0, 1), has_aux=True)(logits, params, key, x)
    grads = jax.lax.pmean(grads, axis_name="p")
    updates, opt_state = optimizer.update(grads, opt_state)
    logits, params = optax.apply_updates((logits, params), updates)

    statistics = jax.lax.pmean(aux["statistics"], axis_name="p")
    key, x = aux["key"], aux["x"]
    S_logits = aux["S_logits"]

    auxiliary_data = {"statistics": statistics,
                      "S_logits": S_logits,
                     }
    return logits, params, opt_state, key, x, auxiliary_data

for i in range(2000):
    logits, params, opt_state, keys, x, aux = update(logits, params, opt_state, keys, x)
    aux = jax.tree_map(lambda x: x[0], aux)
    E, E2_mean, F, F2_mean, S, S2_mean = \
            aux["statistics"]["E_mean"], aux["statistics"]["E2_mean"], \
            aux["statistics"]["F_mean"], aux["statistics"]["F2_mean"], \
            aux["statistics"]["S_mean"], aux["statistics"]["S2_mean"]
    E_std = jnp.sqrt(E2_mean - E**2)
    F_std = jnp.sqrt(F2_mean - F**2)
    S_std = jnp.sqrt(S2_mean - S**2)
    S_logits = aux["S_logits"]
    print("iter: %04d" % i,
            "F:", F, "F_std:", F_std, 
            "E:", E, "E_std:", E_std, 
            "S:", S, "S_std:", S_std, "S_logits:", S_logits)
