# CoulombGas

[![Build Status](https://github.com/fermiflow/CoulombGas/actions/workflows/tests.yml/badge.svg)](https://github.com/fermiflow/CoulombGas/actions)
[![Paper](https://img.shields.io/badge/paper-arXiv:2201.03156-B31B1B.svg)](https://arxiv.org/abs/2201.03156)

This code implements the neural canonical transformation approach to the thermodynamic properties of uniform electron gas. Building on [JAX](https://github.com/google/jax), it utilizes (both forward- and reverse-mode) automatic differentiation and the [pmap](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap) mechanism to achieve a large-scale single-program multiple-data (SPMD) training on multiple GPUs.

## Requirements

- [JAX](https://github.com/google/jax) with Nvidia GPU support
- A handful of GPUs. The more the better :P
- [haiku](https://github.com/deepmind/dm-haiku)
- [optax](https://github.com/deepmind/optax)
- To analytically computing the thermal entropy of a non-interacting Fermi gas in the canonical ensemble based on [arbitrary-precision arithmetic](https://en.wikipedia.org/wiki/Arbitrary-precision_arithmetic), we have used the python library [mpmath](https://mpmath.org).

## Demo run

To start, try running the following commands to launch a training of 13 spin-polarized electrons in 2D with the dimensionless density parameter 10.0 and (reduced) temperature 0.15 on 8 GPUs:

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python main.py --n 13 --dim 2 --rs 10.0 --Theta 0.15 --Emax 25 --sr --batch 4096 --num_devices 8 --acc_steps 2
```

Note that we effectively sample a batch of totally **8192** samples in each training step. However, such a batch size will result in too large a memory consumption to be accommodated by 8 GPUs. To overcome this problem, we choose to split the batch into two equal pieces, and accumulate the gradient and various observables for each piece in two sequential substeps. In other words, the argument `batch` in the command above actually stands for the batch per accumulation step.

If you have only, say, 4 GPUs, you can set `batch`, `num_devices`, `acc_steps` to be 2048, 4 and 4 respectively to launch the same training process, at the expense of doubling the running time. The GPU hours are nevertheless the same.

For the detail meanings of other command line arguments, run

```shell
python main.py --help
```

or directly refer to the source code.

## Trained model and data

A training process from complete scratch actually contains two stages. In the first stage, a variational autoregressive network is **pretrained** to approximate the Boltzmann distribution of the corresponding non-interacting electron gas. The resulting model can be saved and then loaded later. In fact, we have provided such a [model file](https://github.com/fermiflow/CoulombGas/blob/master/data/freefermion/pretraining/n_13_dim_2_Theta_0.15_Emax_25_twist_0.250_0.250/nlayers_2_modelsize_16_nheads_4_nhidden_32_damping_0.00100_maxnorm_0.00100_batch_8192/params_van.pkl) for the parameter settings of the last section for your convenience, so you can quickly get a feeling of the second stage of training the truly interacting system of our interest. We encourage you to remove the file to pretrain the model by yourself; it is actually much faster than the training in the second stage.

To facilitate further developments, we also provide the trained models and logged data for various calculations in the paper, which are located in the [data](https://github.com/fermiflow/CoulombGas/tree/master/data) directory.

## To cite

```
@misc{xie2022mast,
      title={$m^\ast$ of two-dimensional electron gas: a neural canonical transformation study}, 
      author={Hao Xie and Linfeng Zhang and Lei Wang},
      year={2022},
      eprint={2201.03156},
      archivePrefix={arXiv},
      primaryClass={cond-mat.stat-mech}
}
```

