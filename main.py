import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

key = jax.random.PRNGKey(1)

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
parser.add_argument("--mc_therm", type=int, default=10, help="MCMC thermalization steps")
parser.add_argument("--mc_steps", type=int, default=50, help="MCMC update steps")
parser.add_argument("--mc_stddev", type=float, default=0.1, help="standard deviation of the Gaussian proposal in MCMC update")

#training parameters
parser.add_argument("--batch", type=int, default=1024, help="batch size")
#parser.add_argument("--k_steps", type=int, default=1, help="accumulation steps")
#parser.add_argument("--damping", type=float, default=1e-3, help="damping")
#parser.add_argument("--max_norm", type=float, default=1e-3, help="gradnorm")
#parser.add_argument("--use_sparse_solver", action='store_true',  help="")
parser.add_argument("--epochs", type=int, default=10000, help="epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

#model parameters
#parser.add_argument("--steps", type=int, default=2, help="FermiNet: steps")
parser.add_argument("--depth", type=int, default=2, help="FermiNet: depth")
parser.add_argument("--spsize", type=int, default=16, help="FermiNet: spsize")
parser.add_argument("--tpsize", type=int, default=16, help="FermiNet: tpsize")

args = parser.parse_args()

n, dim = args.n, args.dim
if dim == 3:
    L = (4/3*jnp.pi*n)**(1/3)
    beta = 1 / ((4.5*jnp.pi)**(2/3) * args.Theta)
elif dim == 2:
    L = jnp.sqrt(jnp.pi*n)
    beta = 1/ (4 * args.Theta)
print("n = %d, dim = %d, L = %f" % (n, dim, L))

####################################################################################

print("Initialize many-body state distribution...")

from orbitals import manybody_orbitals
manybody_indices, manybody_Es = manybody_orbitals(n, dim, args.Ecut)
manybody_indices, manybody_Es = jnp.array(manybody_indices), jnp.array(manybody_Es)
print("manybody_indices.shape:", manybody_indices.shape)
logits = - beta * manybody_Es * (2*jnp.pi/L)**2
logits -= jax.scipy.special.logsumexp(logits)
print("beta = %f, Ecut = %f, corresponding delta_logit = %f"
        % (beta, args.Ecut, beta * (2*jnp.pi/L)**2 * args.Ecut))

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
logpsi_grad_laplacian, logp = make_logpsi_logp(flow, manybody_indices, L)

####################################################################################

print("Initialize relevant quantities for Ewald summation...")

from potential import kpoints, Madelung
G = kpoints(dim, args.Gmax)
Vconst = n * args.rs/L * Madelung(dim, args.kappa, G)
print("(scaled) Vconst:", Vconst/(n*args.rs/L))

####################################################################################

print("Generate initial key and coordinate sample...")

num_devices = jax.device_count()
print("Number of GPU devices:", num_devices)
if args.batch % num_devices != 0:
    raise ValueError("Batch size must be divisible by the number of GPU devices. "
                     "Got batch = %d for %d devices now." % (args.batch, num_devices))
batch_per_device = args.batch // num_devices

from utils import shard, replicate

x = jax.random.uniform(key, (num_devices, batch_per_device, n, dim), minval=0., maxval=L)
keys = jax.random.split(key, num_devices)
x, keys = shard(x), shard(keys)
origin_logits, origin_params = logits, params
logits, params = replicate((logits, params), num_devices)
print("keys:", keys, "\nshape:", keys.shape, "\t\ttype:", type(keys))
print("x:", x, "\nshape:", x.shape, "\t\ttype:", type(x))

from VMC import sample_stateindices_and_x
pmap_sample_x = jax.pmap(sample_stateindices_and_x, in_axes=(0, 0, None, 0, 0, None, None, None),
                                   static_broadcasted_argnums=2)
for i in range(args.mc_therm):
    print("---- thermal step %d ----" % (i+1))
    keys, _, x = pmap_sample_x(keys, logits, logp, x, params, args.mc_steps, args.mc_stddev, L)
print("keys:", keys, "\nshape:", keys.shape, "\t\ttype:", type(keys))
print("x:", x, "\nshape:", x.shape, "\t\ttype:", type(x))

####################################################################################

print("Initialize optimizer...")
import optax

optimizer = optax.adam(args.lr)
opt_state = optimizer.init((origin_logits, origin_params))
opt_state = replicate(opt_state, num_devices)

####################################################################################

from VMC import make_loss
loss_fn = make_loss(logp, args.mc_steps, args.mc_stddev,
                    logpsi_grad_laplacian,
                    args.kappa, G, L, args.rs, Vconst, beta)

from functools import partial

@partial(jax.pmap, axis_name="p")
def update(logits, params, opt_state, key, x):
    grads, aux = jax.grad(loss_fn, argnums=(0, 1), has_aux=True)(logits, params, key, x)
    grads = jax.lax.pmean(grads, axis_name="p")
    updates, opt_state = optimizer.update(grads, opt_state)
    logits, params = optax.apply_updates((logits, params), updates)

    key, x = aux["key"], aux["x"]
    auxiliary_data = aux["statistics"]
    return logits, params, opt_state, key, x, auxiliary_data

for i in range(args.epochs):
    logits, params, opt_state, keys, x, aux = update(logits, params, opt_state, keys, x)
    aux = jax.tree_map(lambda x: x[0], aux)
    K, K2_mean, V, V2_mean, E, E2_mean, F, F2_mean, S, S2_mean, S_logits = \
            aux["K_mean"], aux["K2_mean"], aux["V_mean"], aux["V2_mean"], \
            aux["E_mean"], aux["E2_mean"], aux["F_mean"], aux["F2_mean"], \
            aux["S_mean"], aux["S2_mean"], aux["S_logits"]
    K_std = jnp.sqrt((K2_mean - K**2) / args.batch)
    V_std = jnp.sqrt((V2_mean - V**2) / args.batch)
    E_std = jnp.sqrt((E2_mean - E**2) / args.batch)
    F_std = jnp.sqrt((F2_mean - F**2) / args.batch)
    S_std = jnp.sqrt((S2_mean - S**2) / args.batch)
    print("iter: %04d" % i,
            "F:", F, "F_std:", F_std, 
            "E:", E, "E_std:", E_std, 
            "K:", K, "K_std:", K_std, 
            "V:", V, "V_std:", V_std, 
            "S:", S, "S_std:", S_std, "S_logits:", S_logits)
