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
parser.add_argument("--kappa", type=float, default=8, help="screening parameter (in unit of 1/L) in Ewald summation")

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
Vconst = Madelung(dim, args.kappa, G)
print("Vconst:", Vconst)

####################################################################################

print("Generate initial key and coordinate sample...")

from state_sampler import make_softmax_sampler
from VMC import MCMC_thermalize
sampler, _ = make_softmax_sampler(logits)
key, x = MCMC_thermalize(key, args.batch, n, dim, L,
                         sampler, params, logp, args.mc_steps, args.mc_therm)

####################################################################################

from VMC import make_loss
import optax

lr = 1e-2
optimizer = optax.adam(lr)
opt_state = optimizer.init((logits, params))

loss_fn = make_loss(logp, args.mc_steps, logpsi,
          args.kappa, G, Vconst, L, args.rs, beta)

for i in range(500):
    grads, aux = jax.grad(loss_fn, argnums=(0, 1), has_aux=True)(logits, params, key, x)
    key, x = aux["key"], aux["x"]
    updates, opt_state = optimizer.update(grads, opt_state)
    logits, params = optax.apply_updates((logits, params), updates)

    E, E_std, F, F_std, S, S_std, S_logits = \
            aux["E"], aux["E_std"], aux["F"], aux["F_std"], \
            aux["S"], aux["S_std"], aux["S_logits"]
    print("iter: %04d" % i,
            "F:", F, "F_std:", F_std, 
            "E:", E, "E_std:", E_std, 
            "S:", S, "S_std:", S_std, "S_logits:", S_logits)
