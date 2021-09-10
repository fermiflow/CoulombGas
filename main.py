import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

key = jax.random.PRNGKey(42)

import argparse
parser = argparse.ArgumentParser(description="Finite-temperature VMC for homogeneous electron gas")

parser.add_argument("--folder", default="/data1/xieh/CoulombGas/master/", help="the folder to save computed data")

# physical parameters.
parser.add_argument("--n", type=int, default=13, help="total number of electrons")
parser.add_argument("--dim", type=int, default=2, help="spatial dimension")
parser.add_argument("--rs", type=float, default=1.0, help="rs")
parser.add_argument("--Theta", type=float, default=0.05, help="dimensionless temperature T/Ef")

# many-body state distribution: autoregressive transformer.
parser.add_argument("--Emax", type=int, default=25, help="energy cutoff for the single-particle orbitals")
parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
parser.add_argument("--modelsize", type=int, default=32, help="model size")
parser.add_argument("--nheads", type=int, default=4, help="number of heads")
parser.add_argument("--nhidden", type=int, default=48, help="number of hidden dimension of the MLP within transformer layers")

# normalizing flow.
#parser.add_argument("--steps", type=int, default=2, help="FermiNet: steps")
parser.add_argument("--depth", type=int, default=2, help="FermiNet: depth")
parser.add_argument("--spsize", type=int, default=16, help="FermiNet: spsize")
parser.add_argument("--tpsize", type=int, default=16, help="FermiNet: tpsize")

# parameters relevant to th Ewald summation of Coulomb interaction.
parser.add_argument("--Gmax", type=int, default=15, help="k-space cutoff in the ewald summation of Coulomb potential")
parser.add_argument("--kappa", type=int, default=10, help="screening parameter (in unit of 1/L) in Ewald summation")

# MCMC.
parser.add_argument("--mc_therm", type=int, default=10, help="MCMC thermalization steps")
parser.add_argument("--mc_steps", type=int, default=50, help="MCMC update steps")
parser.add_argument("--mc_stddev", type=float, default=0.1, help="standard deviation of the Gaussian proposal in MCMC update")

# optimizer parameters.
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (valid only for adam)")
parser.add_argument("--sr", action='store_true',  help="use the second-order stochastic reconfiguration optimizer")
parser.add_argument("--damping", type=float, default=1e-3, help="damping")
parser.add_argument("--max_norm", type=float, default=1e-3, help="gradnorm maximum")
#parser.add_argument("--use_sparse_solver", action='store_true',  help="")

# training parameters.
parser.add_argument("--batch", type=int, default=1024, help="batch size")
parser.add_argument("--num_devices", type=int, default=8, help="number of GPU devices")
#parser.add_argument("--k_steps", type=int, default=1, help="accumulation steps")
parser.add_argument("--epoch_finished", type=int, default=0, help="number of epochs already finished")
parser.add_argument("--epoch", type=int, default=10000, help="final epoch")

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

print("========== Initialize single-particle orbitals ==========")

from orbitals import sp_orbitals
indices, Es = sp_orbitals(dim)
indices, Es = jnp.array(indices), jnp.array(Es)
Ef = Es[n-1]
print("beta = %f, Ef = %d, Emax = %d, corresponding delta_logit = %f"
        % (beta, Ef, args.Emax, beta * (2*jnp.pi/L)**2 * (args.Emax - Ef)))

indices, Es = indices[Es<=args.Emax], Es[Es<=args.Emax]
num_states = Es.size
print("Number of available single-particle orbitals: %d" % num_states)
from scipy.special import comb
print("Total number of many-body states (%d in %d): %f" % (n, num_states, comb(num_states, n)))

from orbitals import manybody_orbitals
manybody_indices, manybody_Es = manybody_orbitals(n, dim, 8)
manybody_indices, manybody_Es = jnp.array(manybody_indices), jnp.array(manybody_Es)
print("manybody_indices.shape:", manybody_indices.shape)
logits = - beta * manybody_Es * (2*jnp.pi/L)**2
logits -= jax.scipy.special.logsumexp(logits)

E = (manybody_Es * (2*jnp.pi/L)**2 * jnp.exp(logits)).sum()
S = -(logits * jnp.exp(logits)).sum()
F = E - S / beta
print("F:", F, "\tE:", E, "\tS:", S)

####################################################################################

print("========== Initialize many-body state distribution ==========")

import haiku as hk
from autoregressive import Transformer
def forward_fn(state_idx):
    model = Transformer(num_states, args.nlayers, args.modelsize, args.nheads, args.nhidden)
    return model(state_idx[..., None])
van = hk.transform(forward_fn)
state_idx_dummy = jnp.arange(n, dtype=jnp.float64)
params_van = van.init(key, state_idx_dummy)

raveled_params_van, _ = ravel_pytree(params_van)
print("#parameters in the autoregressive model: %d" % raveled_params_van.size)

from freefermion import pretrain
params_van = pretrain(van, params_van, (2*jnp.pi/L)**2 * Es[::-1], beta,
                        n, dim, key, 8192, 2000)
exit(0)

####################################################################################

print("========== Initialize normalizing flow ==========")

from flow import FermiNet
def flow_fn(x):
    model = FermiNet(args.depth, args.spsize, args.tpsize, L)
    return model(x)
flow = hk.transform(flow_fn)
x_dummy = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
params_flow = flow.init(key, x_dummy)

raveled_params_flow, _ = ravel_pytree(params_flow)
print("#parameters in the flow model: %d" % raveled_params_flow.size)

from logpsi import make_logpsi, make_logpsi_grad_laplacian, make_logp, make_quantum_score
logpsi = make_logpsi(flow, manybody_indices, L)
logp = make_logp(logpsi)

####################################################################################

print("Initialize relevant quantities for Ewald summation...")

from potential import kpoints, Madelung
G = kpoints(dim, args.Gmax)
Vconst = n * args.rs/L * Madelung(dim, args.kappa, G)
print("(scaled) Vconst:", Vconst/(n*args.rs/L))

####################################################################################

print("Initialize optimizer...")

import optax
if args.sr:
    from sr import hybrid_fisher_sr
    from softmax import classical_score_fn
    quantum_score_fn = make_quantum_score(logpsi)
    optimizer = hybrid_fisher_sr(classical_score_fn, quantum_score_fn,
            args.damping, args.max_norm)
    print("Optimizer hybrid_fisher_sr: damping = %.5f, max_norm = %.5f." %
            (args.damping, args.max_norm))
else:
    optimizer = optax.adam(args.lr)
    print("Optimizer adam: lr = %.3f." % args.lr)

####################################################################################

import os
import checkpoint
from utils import shard, replicate

path = args.folder + "n_%d_dim_%d_rs_%f_Theta_%f" % (n, dim, args.rs, args.Theta) \
                   + "_Ecut_%.1f" % args.Ecut \
                   + "_depth_%d_spsize_%d_tpsize_%d" % (args.depth, args.spsize, args.tpsize) \
                   + "_Gmax_%d_kappa_%d" % (args.Gmax, args.kappa) \
                   + "_mctherm_%d_mcsteps_%d_mcstddev_%.2f" % (args.mc_therm, args.mc_steps, args.mc_stddev) \
                   + ("_damping_%.5f_maxnorm_%.5f" % (args.damping, args.max_norm)
                        if args.sr else "_lr_%.3f" % args.lr) \
                   + "_batch_%d_ndevices_%d" % (args.batch, args.num_devices)
if not os.path.isdir(path):
    os.makedirs(path)
    print("Create directory: %s" % path)
load_ckpt_filename = checkpoint.ckpt_filename(args.epoch_finished, path)

num_devices = args.num_devices
print("Number of GPU devices:", num_devices)
if num_devices != jax.device_count():
    raise ValueError("Expected %d GPU devices. Got %d." % (num_devices, jax.device_count()))

if os.path.isfile(load_ckpt_filename):
    print("Load checkpoint file: %s" % load_ckpt_filename)
    ckpt = checkpoint.load_checkpoint(load_ckpt_filename)
    keys, x, logits, params, opt_state = \
        ckpt["keys"], ckpt["x"], ckpt["logits"], ckpt["params"], ckpt["opt_state"]
    x, keys = shard(x), shard(keys)
    logits, params = replicate((logits, params), num_devices)
else:
    print("No checkpoint file found. Start from scratch.")

    opt_state = optimizer.init((logits, params))

    print("Initialize key and coordinate samples...")

    if args.batch % num_devices != 0:
        raise ValueError("Batch size must be divisible by the number of GPU devices. "
                         "Got batch = %d for %d devices now." % (args.batch, num_devices))
    batch_per_device = args.batch // num_devices

    x = jax.random.uniform(key, (num_devices, batch_per_device, n, dim), minval=0., maxval=L)
    keys = jax.random.split(key, num_devices)
    x, keys = shard(x), shard(keys)
    logits, params = replicate((logits, params), num_devices)

    from VMC import sample_stateindices_and_x
    pmap_sample_x = jax.pmap(sample_stateindices_and_x, in_axes=(0, 0, None, 0, 0, None, None, None),
                                       static_broadcasted_argnums=2)
    for i in range(args.mc_therm):
        print("---- thermal step %d ----" % (i+1))
        keys, _, x = pmap_sample_x(keys, logits, logp, x, params, args.mc_steps, args.mc_stddev, L)
    print("keys shape:", keys.shape, "\t\ttype:", type(keys))
    print("x shape:", x.shape, "\t\ttype:", type(x))

####################################################################################

logpsi_grad_laplacian = make_logpsi_grad_laplacian(logpsi)

from VMC import make_loss
loss_fn = make_loss(logpsi_grad_laplacian, args.kappa, G, L, args.rs, Vconst, beta)

from functools import partial

@partial(jax.pmap, axis_name="p",
        in_axes=(0, 0, None, 0, 0),
        out_axes=(0, 0, None, 0, 0, 0, 0, 0))
def update(logits, params, opt_state, key, x):

    key, state_indices, x = sample_stateindices_and_x(key, logits,
                                    logp, x, params, args.mc_steps, args.mc_stddev, L)
    print("Sampled state indices and electron coordinates.")

    grads, aux = jax.grad(loss_fn, argnums=(0, 1), has_aux=True)(logits, params, state_indices, x)
    grads = jax.lax.pmean(grads, axis_name="p")

    updates, opt_state = optimizer.update(grads, opt_state,
                            params=(logits, params, state_indices, x) if args.sr else None)
    logits, params = optax.apply_updates((logits, params), updates)

    logpsi, Eloc_real = aux["logpsi"], aux["Eloc_real"]
    auxiliary_data = aux["statistics"]
    return logits, params, opt_state, key, x, auxiliary_data, logpsi, Eloc_real

log_filename = os.path.join(path, "data.txt")
f = open(log_filename, "w" if args.epoch_finished == 0 else "a",
            buffering=1, newline="\n")

for i in range(args.epoch_finished + 1, args.epoch + 1):
    logits, params, opt_state, keys, x, aux, logpsi, Eloc_real \
        = update(logits, params, opt_state, keys, x)

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

    # Note the quantities with energy dimension obtained above are in units of Ry/rs^2.
    print("iter: %04d" % i,
            "F:", F/args.rs**2, "F_std:", F_std/args.rs**2,
            "E:", E/args.rs**2, "E_std:", E_std/args.rs**2,
            "K:", K/args.rs**2, "K_std:", K_std/args.rs**2,
            "V:", V/args.rs**2, "V_std:", V_std/args.rs**2,
            "S:", S, "S_std:", S_std, "S_logits:", S_logits)
    f.write( ("%6d" + "  %.6f"*11 + "\n") % (i, F/args.rs**2, F_std/args.rs**2,
                                                E/args.rs**2, E_std/args.rs**2,
                                                K/args.rs**2, K_std/args.rs**2,
                                                V/args.rs**2, V_std/args.rs**2,
                                                S, S_std, S_logits) )

    if i % 100 == 0:
        ckpt = {"keys": keys, "x": x,
                "logits": jax.tree_map(lambda x: x[0], logits),
                "params": jax.tree_map(lambda x: x[0], params),
                "opt_state": opt_state
               }
        save_ckpt_filename = checkpoint.ckpt_filename(i, path)
        checkpoint.save_checkpoint(ckpt, save_ckpt_filename)
        print("Save checkpoint file: %s" % save_ckpt_filename)

    """
    logpsi_real = logpsi.real.reshape(-1)
    Eloc_real = Eloc_real.reshape(-1)
    print("Eloc_real max: (%f, %d)" % (Eloc_real.max(), Eloc_real.argmax()),
          "min: (%f, %d)" % (Eloc_real.min(), Eloc_real.argmin()),
          "mean:", Eloc_real.mean(), "std:", Eloc_real.std())
    print("logpsi_real max: (%f, %d)" % (logpsi_real.max(), logpsi_real.argmax()),
          "min: (%f, %d)" % (logpsi_real.min(), logpsi_real.argmin()),
          "mean:", logpsi_real.mean(), "std:", logpsi_real.std())
    """

f.close()
