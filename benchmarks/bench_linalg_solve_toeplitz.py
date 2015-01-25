"""Benchmark the solve_toeplitz solver (Levinson recursion)
"""
from __future__ import division, absolute_import, print_function
from .common import Benchmark, SimpleTimer

import numpy as np
from numpy.testing import assert_array_almost_equal
import scipy.linalg


class SolveToeplitz(Benchmark):
    group_by = {
        'gen_solve_toeplitz': ['row', 'row', 'col']
    }

    @classmethod
    def gen_solve_toeplitz(cls):
        random = np.random.RandomState(1234)

        def track_solve_toeplitz(self, dtype, n, soltype):
            n = int(n)
            dtype = np.dtype(dtype)

            # Sample a random Toeplitz matrix representation and rhs.
            c = random.randn(n)
            r = random.randn(n)
            y = random.randn(n)
            if dtype == np.complex128:
                c = c + 1j*random.rand(n)
                r = r + 1j*random.rand(n)
                y = y + 1j*random.rand(n)

            if soltype == 'toeplitz':
                tm_gen = [0]
                tm_toep = SimpleTimer(min_iter=10)
            else:
                tm_gen = SimpleTimer(min_iter=10)
                tm_toep = [0]

            # generic solver
            for _ in tm_gen:
                T = scipy.linalg.toeplitz(c, r=r)
                x_generic = scipy.linalg.solve(T, y)

            # toeplitz-specific solver
            for _ in tm_toep:
                x_toeplitz = scipy.linalg.solve_toeplitz((c, r), y)

            # Check that the solutions are the sameself.
            assert_array_almost_equal(x_generic, x_toeplitz)

            if soltype == 'toeplitz':
                return tm_toep.timing
            else:
                return tm_gen.timing

        for dtype in ('float64', 'complex128'):
            for n in (100, 300, 1000):
                for soltype in ['toeplitz', 'generic']:
                    yield track_solve_toeplitz, dtype, str(n), soltype
