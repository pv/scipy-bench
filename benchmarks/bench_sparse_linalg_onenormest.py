"""Compare the speed of exact one-norm calculation vs. its estimation.
"""
from __future__ import division, print_function, absolute_import
from .common import Benchmark

import time

import numpy as np

import scipy.sparse.linalg


class BenchmarkOneNormEst(Benchmark):
    group_by = {
        'gen_onenormest': ['row', 'col'],
    }

    @classmethod
    def gen_onenormest(cls):
        def track(self, n, soltype):
            n = int(n)
            np.random.seed(1234)
            nrepeats = 100
            shape = (n, n)

            # Sample the matrices.
            matrices = []
            for i in range(nrepeats):
                M = np.random.randn(*shape)
                matrices.append(M)

            if soltype == 'exact':
                # Get the exact values of one-norms of squares.
                tm_start = time.clock()
                for M in matrices:
                    M2 = M.dot(M)
                    scipy.sparse.linalg.matfuncs._onenorm(M)
                tm_end = time.clock()
                tm_exact = tm_end - tm_start
                return tm_exact
            elif soltype == 'estimate':
                # Get the estimates of one-norms of squares.
                tm_start = time.clock()
                for M in matrices:
                    scipy.sparse.linalg.matfuncs._onenormest_matrix_power(M, 2)
                tm_end = time.clock()
                tm_estimate = tm_end - tm_start
                return tm_estimate

        for n in (2, 3, 5, 10, 30, 100, 300, 500, 1000):
            for soltype in ['exact', 'estimate']:
                yield track, str(n), soltype
