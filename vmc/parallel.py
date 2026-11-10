"""
MPI helpers. The walkers are independent Markov chains, so they are split over the ranks and
the only communication is a few small reductions per optimization iteration (energy
statistics and the SR matrices). Without mpi4py, or with a single process, every helper
reduces to the identity and the serial code path is the same code.

Functions marked (collective) must be called by all ranks.
"""
import os

# The matrices are tiny, so BLAS threads only add overhead, and with one rank per core they
# oversubscribe the node. This must run before numpy is imported anywhere; override by
# setting the variables yourself.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
except (ImportError, RuntimeError):      # no mpi4py, or mpi4py without a usable MPI library
    COMM = None

RANK = COMM.Get_rank() if COMM is not None else 0
SIZE = COMM.Get_size() if COMM is not None else 1
ROOT = RANK == 0


def n_local(n_total):
    """Number of walkers on this rank when n_total walkers are shared as evenly as possible."""
    if n_total < SIZE:
        raise ValueError(f'N_WALKERS = {n_total} is smaller than the number of MPI ranks ({SIZE}).')
    return n_total // SIZE + (RANK < n_total % SIZE)


def sum_all(x):
    """(collective) Sum of a scalar or numpy array over all ranks, returned on every rank."""
    return x if COMM is None else COMM.allreduce(x)


def concat_all(x):
    """(collective) Concatenate the per-rank arrays x of shape (W_local, ...) along axis 0."""
    return x if COMM is None else np.concatenate(COMM.allgather(x))


def broadcast(x):
    """(collective) The value of x on rank 0, on every rank."""
    return x if COMM is None else COMM.bcast(x, root=0)


def mean_all(x):
    """(collective) Mean over axis 0 of x, pooling the samples of all ranks."""
    return sum_all(x.sum(axis=0)) / sum_all(len(x))