import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

print("jax.__version__:", jax.__version__)
key = jax.random.PRNGKey(42)

import argparse
parser = argparse.ArgumentParser(description="Finite-temperature VMC for the homogeneous electron gas")

parser.add_argument("--folder", default="data/", help="the folder to save data")

# physical parameters.
parser.add_argument("--n", type=int, default=29, help="total number of electrons")
parser.add_argument("--dim", type=int, default=2, help="spatial dimension")
parser.add_argument("--rs", type=float, default=10.0, help="rs")
parser.add_argument("--Theta", type=float, default=0.15, help="dimensionless temperature T/Ef")

# many-body state distribution: autoregressive transformer.
parser.add_argument("--Emax", type=int, default=25, help="energy cutoff for the single-particle orbitals")
parser.add_argument("--nlayers", type=int, default=2, help="CausalTransformer: number of layers")
parser.add_argument("--modelsize", type=int, default=16, help="CausalTransformer: embedding dimension")
parser.add_argument("--nheads", type=int, default=4, help="CausalTransformer:number of heads")
parser.add_argument("--nhidden", type=int, default=32, help="CausalTransformer: number of hidden units of the MLP within each layer")

# normalizing flow.
parser.add_argument("--depth", type=int, default=2, help="FermiNet: network depth")
parser.add_argument("--spsize", type=int, default=16, help="FermiNet: single-particle feature size")
parser.add_argument("--tpsize", type=int, default=16, help="FermiNet: two-particle feature size")

# parameters relevant to th Ewald summation of Coulomb interaction.
parser.add_argument("--Gmax", type=int, default=15, help="k-space cutoff in the Ewald summation of Coulomb potential")
parser.add_argument("--kappa", type=int, default=10, help="screening parameter (in unit of 1/L) in Ewald summation")

# MCMC.
parser.add_argument("--mc_therm", type=int, default=10, help="MCMC thermalization steps")
parser.add_argument("--mc_steps", type=int, default=50, help="MCMC update steps")
parser.add_argument("--mc_stddev", type=float, default=0.1, help="standard deviation of the Gaussian proposal in MCMC update")

# technical miscellaneous
parser.add_argument("--hutchinson", action='store_true',  help="use Hutchinson's trick to compute the laplacian")

# optimizer parameters.
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (valid only for adam)")
parser.add_argument("--sr", action='store_true',  help="use the second-order stochastic reconfiguration optimizer")
parser.add_argument("--damping", type=float, default=1e-3, help="damping")
parser.add_argument("--max_norm", type=float, default=1e-3, help="gradnorm maximum")

# training parameters.
parser.add_argument("--batch", type=int, default=2048, help="batch size (per single gradient accumulation step)")
parser.add_argument("--num_devices", type=int, default=8, help="number of GPU devices")
parser.add_argument("--acc_steps", type=int, default=4, help="gradient accumulation steps")
parser.add_argument("--epoch_finished", type=int, default=0, help="number of epochs already finished")
parser.add_argument("--epoch", type=int, default=3000, help="final epoch")

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

print("\n========== Initialize single-particle orbitals ==========")

from orbitals import sp_orbitals
sp_indices, Es = sp_orbitals(dim, args.Emax)
sp_indices, Es = jnp.array(sp_indices), jnp.array(Es)
Ef = Es[n-1]
print("beta = %f, Ef = %d, Emax = %d, corresponding delta_logit = %f"
        % (beta, Ef, args.Emax, beta * (2*jnp.pi/L)**2 * (args.Emax - Ef)))

sp_indices, Es = sp_indices[::-1], Es[::-1]
num_states = Es.size
print("Number of available single-particle orbitals: %d" % num_states)
from scipy.special import comb
print("Total number of many-body states (%d in %d): %f" % (n, num_states, comb(num_states, n)))

####################################################################################

print("\n========== Initialize many-body state distribution ==========")

import haiku as hk
from autoregressive import Transformer
def forward_fn(state_idx):
    model = Transformer(num_states, args.nlayers, args.modelsize, args.nheads, args.nhidden)
    return model(state_idx)
van = hk.transform(forward_fn)
state_idx_dummy = sp_indices[-n:].astype(jnp.float64)
params_van = van.init(key, state_idx_dummy)

raveled_params_van, _ = ravel_pytree(params_van)
print("#parameters in the autoregressive model: %d" % raveled_params_van.size)

from sampler import make_autoregressive_sampler, make_classical_score
sampler, log_prob_novmap = make_autoregressive_sampler(van, sp_indices, n, num_states)
log_prob = jax.vmap(log_prob_novmap, (None, 0), 0)

####################################################################################

print("\n========== Pretraining ==========")

# Pretraining parameters for the free-fermion model.
pre_lr = 1e-3
pre_sr, pre_damping, pre_maxnorm = True, 0.001, 0.001
pre_batch = 8192

freefermion_path = args.folder + "freefermion/pretraining/" \
                + "n_%d_dim_%d_Theta_%f_Emax_%d/" % (n, dim, args.Theta, args.Emax) \
                + "nlayers_%d_modelsize_%d_nheads_%d_nhidden_%d" % \
                    (args.nlayers, args.modelsize, args.nheads, args.nhidden) \
                + ("_damping_%.5f_maxnorm_%.5f" % (pre_damping, pre_maxnorm)
                    if pre_sr else "_lr_%.3f" % pre_lr) \
                + "_batch_%d" % pre_batch

import os
if not os.path.isdir(freefermion_path):
    os.makedirs(freefermion_path)
    print("Create freefermion directory: %s" % freefermion_path)

import checkpoint
pretrained_model_filename = checkpoint.pretrained_model_filename(freefermion_path)
if os.path.isfile(pretrained_model_filename):
    print("Load pretrained free-fermion model parameters from file: %s" % pretrained_model_filename)
    params_van = checkpoint.load_data(pretrained_model_filename)
else:
    print("No pretrained free-fermion model found. Initialize parameters from scratch...")
    from freefermion.pretraining import pretrain
    params_van = pretrain(van, params_van,
                          n, dim, args.Theta, args.Emax,
                          freefermion_path, key,
                          pre_lr, pre_sr, pre_damping, pre_maxnorm,
                          pre_batch, epoch=5000)
    print("Initialization done. Save the model to file: %s" % pretrained_model_filename)
    checkpoint.save_data(params_van, pretrained_model_filename)

####################################################################################

print("\n========== Initialize normalizing flow ==========")

from flow import FermiNet
def flow_fn(x):
    model = FermiNet(args.depth, args.spsize, args.tpsize, L)
    return model(x)
flow = hk.transform(flow_fn)
x_dummy = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
params_flow = flow.init(key, x_dummy)

raveled_params_flow, _ = ravel_pytree(params_flow)
print("#parameters in the flow model: %d" % raveled_params_flow.size)

from logpsi import make_logpsi, make_logphi_logjacdet, make_logpsi_grad_laplacian, \
                   make_logp, make_quantum_score
logpsi_novmap = make_logpsi(flow, sp_indices, L)
logphi, logjacdet = make_logphi_logjacdet(flow, sp_indices, L)
logp = make_logp(logpsi_novmap)

####################################################################################

print("\n========== Initialize relevant quantities for Ewald summation ==========")

from potential import kpoints, Madelung
G = kpoints(dim, args.Gmax)
Vconst = n * args.rs/L * Madelung(dim, args.kappa, G)
print("(scaled) Vconst:", Vconst/(n*args.rs/L))

####################################################################################

print("\n========== Initialize optimizer ==========")

import optax
if args.sr:
    classical_score_fn = make_classical_score(log_prob_novmap)
    quantum_score_fn = make_quantum_score(logpsi_novmap)
    from sr import hybrid_fisher_sr
    fishers_fn, optimizer = hybrid_fisher_sr(classical_score_fn, quantum_score_fn,
            args.damping, args.max_norm)
    print("Optimizer hybrid_fisher_sr: damping = %.5f, max_norm = %.5f." %
            (args.damping, args.max_norm))
else:
    optimizer = optax.adam(args.lr)
    print("Optimizer adam: lr = %.3f." % args.lr)

####################################################################################

print("\n========== Checkpointing ==========")

from utils import shard, replicate

path = args.folder + "n_%d_dim_%d_rs_%f_Theta_%f" % (n, dim, args.rs, args.Theta) \
                   + "_Emax_%d" % args.Emax \
                   + "_nlayers_%d_modelsize_%d_nheads_%d_nhidden_%d" % \
                      (args.nlayers, args.modelsize, args.nheads, args.nhidden) \
                   + "_depth_%d_spsize_%d_tpsize_%d" % \
                      (args.depth, args.spsize, args.tpsize) \
                   + "_Gmax_%d_kappa_%d" % (args.Gmax, args.kappa) \
                   + "_mctherm_%d_mcsteps_%d_mcstddev_%.2f" % (args.mc_therm, args.mc_steps, args.mc_stddev) \
                   + ("_hutchinson" if args.hutchinson else "") \
                   + ("_damping_%.5f_maxnorm_%.5f" % (args.damping, args.max_norm)
                        if args.sr else "_lr_%.3f" % args.lr) \
                   + "_batch_%d_ndevices_%d_accsteps_%d" % (args.batch, args.num_devices, args.acc_steps)
if not os.path.isdir(path):
    os.makedirs(path)
    print("Create directory: %s" % path)
load_ckpt_filename = checkpoint.ckpt_filename(args.epoch_finished, path)

num_devices = args.num_devices
print("Number of GPU devices:", num_devices)
if num_devices != jax.device_count():
    raise ValueError("Expected %d GPU devices. Got %d." % (num_devices, jax.device_count()))

from VMC import sample_stateindices_and_x

if os.path.isfile(load_ckpt_filename):
    print("Load checkpoint file: %s" % load_ckpt_filename)
    ckpt = checkpoint.load_data(load_ckpt_filename)
    keys, x, params_van, params_flow, opt_state = \
        ckpt["keys"], ckpt["x"], ckpt["params_van"], ckpt["params_flow"], ckpt["opt_state"]
    x, keys = shard(x), shard(keys)
    params_van, params_flow = replicate((params_van, params_flow), num_devices)
else:
    print("No checkpoint file found. Start from scratch.")

    opt_state = optimizer.init((params_van, params_flow))

    print("Initialize key and coordinate samples...")

    if args.batch % num_devices != 0:
        raise ValueError("Batch size must be divisible by the number of GPU devices. "
                         "Got batch = %d for %d devices now." % (args.batch, num_devices))
    batch_per_device = args.batch // num_devices

    x = jax.random.uniform(key, (num_devices, batch_per_device, n, dim), minval=0., maxval=L)
    keys = jax.random.split(key, num_devices)
    x, keys = shard(x), shard(keys)
    params_van, params_flow = replicate((params_van, params_flow), num_devices)

    for i in range(args.mc_therm):
        print("---- thermal step %d ----" % (i+1))
        keys, _, x, accept_rate = sample_stateindices_and_x(keys,
                                   sampler, params_van,
                                   logp, x, params_flow,
                                   args.mc_steps, args.mc_stddev, L)
    print("keys shape:", keys.shape, "\t\ttype:", type(keys))
    print("x shape:", x.shape, "\t\ttype:", type(x))

####################################################################################

print("\n========== Training ==========")

logpsi, logpsi_grad_laplacian = \
        make_logpsi_grad_laplacian(logpsi_novmap, hutchinson=args.hutchinson,
                                   logphi=logphi, logjacdet=logjacdet)

from VMC import make_loss
observable_and_lossfn = make_loss(log_prob, logpsi, logpsi_grad_laplacian,
                                  args.kappa, G, L, args.rs, Vconst, beta)

from functools import partial

@partial(jax.pmap, axis_name="p",
        in_axes=(0, 0, None, 0, 0, 0, 0, 0, 0, 0, None) if args.sr else (0, 0, None, 0, 0, 0, None, None, None, None),
        out_axes=(0, 0, None, 0, 0, 0, 0, 0) if args.sr else (0, 0, None, 0, 0, None, None, None),
        static_broadcasted_argnums=10 if args.sr else (7, 8, 9, 10),
        donate_argnums=(3, 4))
def update(params_van, params_flow, opt_state, state_indices, x, key, grads_acc,
        classical_fisher_acc, quantum_fisher_acc, quantum_score_mean_acc, final_step):

    data, classical_lossfn, quantum_lossfn = observable_and_lossfn(
            params_van, params_flow, state_indices, x, key)

    grad_params_van = jax.grad(classical_lossfn)(params_van)
    grad_params_flow = jax.grad(quantum_lossfn)(params_flow)
    grads = grad_params_van, grad_params_flow
    grads = jax.lax.pmean(grads, axis_name="p")
    grads_acc = jax.tree_multimap(lambda acc, i: acc + i, grads_acc, grads)

    if args.sr:
        classical_fisher, quantum_fisher, quantum_score_mean = fishers_fn(params_van, params_flow, state_indices, x)
        classical_fisher_acc += classical_fisher
        quantum_fisher_acc += quantum_fisher
        quantum_score_mean_acc += quantum_score_mean

    if final_step:
        grads_acc, classical_fisher_acc, quantum_fisher_acc, quantum_score_mean_acc = \
                jax.tree_map(lambda acc: acc / args.acc_steps,
                             (grads_acc, classical_fisher_acc, quantum_fisher_acc, quantum_score_mean_acc))
        updates, opt_state = optimizer.update(grads_acc, opt_state,
                                params=(classical_fisher_acc, quantum_fisher_acc, quantum_score_mean_acc) if args.sr else None)
        params_van, params_flow = optax.apply_updates((params_van, params_flow), updates)

    return params_van, params_flow, opt_state, data, grads_acc, \
            classical_fisher_acc, quantum_fisher_acc, quantum_score_mean_acc

log_filename = os.path.join(path, "data.txt")
f = open(log_filename, "w" if args.epoch_finished == 0 else "a",
            buffering=1, newline="\n")

for i in range(args.epoch_finished + 1, args.epoch + 1):

    grads_acc = jax.tree_map(jnp.zeros_like, (params_van, params_flow))
    grads_acc = shard(grads_acc)
    if args.sr:
        classical_fisher_acc = jnp.zeros((raveled_params_van.size, raveled_params_van.size))
        classical_fisher_acc = replicate(classical_fisher_acc, num_devices)
        quantum_fisher_acc = jnp.zeros((raveled_params_flow.size, raveled_params_flow.size))
        quantum_fisher_acc = replicate(quantum_fisher_acc, num_devices)
        quantum_score_mean_acc = jnp.zeros(raveled_params_flow.size)
        quantum_score_mean_acc = replicate(quantum_score_mean_acc, num_devices)
    else:
        classical_fisher_acc = quantum_fisher_acc = quantum_score_mean_acc = None
    accept_rate_acc = shard(jnp.zeros(num_devices))

    for acc in range(args.acc_steps):
        keys, state_indices, x, accept_rate = sample_stateindices_and_x(keys,
                                               sampler, params_van,
                                               logp, x, params_flow,
                                               args.mc_steps, args.mc_stddev, L)
        accept_rate_acc += accept_rate
        final_step = (acc == args.acc_steps - 1)

        params_van, params_flow, opt_state, data, grads_acc, \
        classical_fisher_acc, quantum_fisher_acc, quantum_score_mean_acc \
            = update(params_van, params_flow, opt_state, state_indices, x, keys, grads_acc,
                     classical_fisher_acc, quantum_fisher_acc, quantum_score_mean_acc, final_step)

        data = jax.tree_map(lambda x: x[0], data)
        if acc == 0:
            data_acc = data
        else:
            data_acc = jax.tree_multimap(lambda acc, i: acc + i, data_acc, data)

    accept_rate = accept_rate_acc[0] / args.acc_steps
    data = jax.tree_map(lambda acc: acc / args.acc_steps, data_acc)
    K, K2_mean, V, V2_mean, E, E2_mean, F, F2_mean, S, S2_mean = \
            data["K_mean"], data["K2_mean"], data["V_mean"], data["V2_mean"], \
            data["E_mean"], data["E2_mean"], data["F_mean"], data["F2_mean"], \
            data["S_mean"], data["S2_mean"]
    K_std = jnp.sqrt((K2_mean - K**2) / (args.batch*args.acc_steps))
    V_std = jnp.sqrt((V2_mean - V**2) / (args.batch*args.acc_steps))
    E_std = jnp.sqrt((E2_mean - E**2) / (args.batch*args.acc_steps))
    F_std = jnp.sqrt((F2_mean - F**2) / (args.batch*args.acc_steps))
    S_std = jnp.sqrt((S2_mean - S**2) / (args.batch*args.acc_steps))

    # Note the quantities with energy dimension obtained above are in units of Ry/rs^2.
    print("iter: %04d" % i,
            "F:", F/args.rs**2, "F_std:", F_std/args.rs**2,
            "E:", E/args.rs**2, "E_std:", E_std/args.rs**2,
            "K:", K/args.rs**2, "K_std:", K_std/args.rs**2,
            "V:", V/args.rs**2, "V_std:", V_std/args.rs**2,
            "S:", S, "S_std:", S_std,
            "accept_rate:", accept_rate)
    f.write( ("%6d" + "  %.6f"*10 + "  %.4f" + "\n") % (i,
                                                F/args.rs**2, F_std/args.rs**2,
                                                E/args.rs**2, E_std/args.rs**2,
                                                K/args.rs**2, K_std/args.rs**2,
                                                V/args.rs**2, V_std/args.rs**2,
                                                S, S_std, accept_rate) )

    if i % 100 == 0:
        ckpt = {"keys": keys, "x": x,
                "params_van": jax.tree_map(lambda x: x[0], params_van),
                "params_flow": jax.tree_map(lambda x: x[0], params_flow),
                "opt_state": opt_state
               }
        save_ckpt_filename = checkpoint.ckpt_filename(i, path)
        checkpoint.save_data(ckpt, save_ckpt_filename)
        print("Save checkpoint file: %s" % save_ckpt_filename)

f.close()
