"""
Check the speed of the conjugate gradient solver.
"""
from __future__ import division, absolute_import, print_function
from .common import Benchmark, measure

import time

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from scipy import linalg, sparse
import scipy.sparse.linalg


def _create_sparse_poisson1d(n):
    # Make Gilbert Strang's favorite matrix
    # http://www-math.mit.edu/~gs/PIX/cupcakematrix.jpg
    P1d = sparse.diags([[-1]*(n-1), [2]*n, [-1]*(n-1)], [-1, 0, 1])
    assert_equal(P1d.shape, (n, n))
    return P1d


def _create_sparse_poisson2d(n):
    P1d = _create_sparse_poisson1d(n)
    P2d = sparse.kronsum(P1d, P1d)
    assert_equal(P2d.shape, (n*n, n*n))
    return P2d


class Bench(Benchmark):
    group_by = {
        'gen_cg': ['row', 'col'],
        'gen_spsolve': ['row', 'col'],
    }

    @classmethod
    def _gen_call(cls, name):
        def track(self, n, soltype):
            n = int(n)
            dense_is_active = (n**2 < 600)
            sparse_is_active = (n**2 < 20000)

            if not dense_is_active and not sparse_is_active:
                return np.nan

            b = np.ones(n*n)
            P_sparse = _create_sparse_poisson2d(n)
            repeats = 100

            # Optionally use the generic dense solver.
            if soltype == 'dense':
                if not dense_is_active:
                    return np.nan

                P_dense = P_sparse.A
                tm_start = time.clock()
                for i in range(repeats):
                    x_dense = linalg.solve(P_dense, b)
                tm_end = time.clock()
                tm = tm_end - tm_start
            else:
                if not sparse_is_active:
                    return np.nan

                # Optionally use the sparse conjugate gradient solver.
                if name == 'spsolve':
                    solver = getattr(sparse.linalg, name)
                    tm_start = time.clock()
                    for i in range(repeats):
                        x_sparse = solver(P_sparse, b)
                    tm_end = time.clock()
                else:
                    solver = getattr(sparse.linalg, name)
                    tm_start = time.clock()
                    for i in range(repeats):
                        x_sparse, info = solver(P_sparse, b)
                    tm_end = time.clock()
                tm = tm_end - tm_start

            return tm

        track.__name__ = "track_" + name

        for n in 4, 6, 10, 16, 25, 40, 64, 100, 160, 250, 400, 640, 1000, 1600:
            for soltype in ['dense', 'sparse']:
                yield track, str(n), soltype

    @classmethod
    def gen_cg(cls):
        for func, n, soltype in cls._gen_call("cg"):
            yield func, n, soltype

    @classmethod
    def gen_spsolve(cls):
        for func, n, soltype in cls._gen_call("spsolve"):
            yield func, n, soltype


if __name__ == '__main__':
    Tester().bench()
