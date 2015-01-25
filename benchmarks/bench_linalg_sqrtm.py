""" Benchmark linalg.sqrtm for various blocksizes.

"""
from __future__ import division, absolute_import, print_function
from .common import Benchmark, measure

import numpy as np
from numpy.testing import assert_allclose
import scipy.linalg


class Sqrtm(Benchmark):
    group_by = {
        'gen_sqrtm': ['row', 'row', 'col']
    }

    @classmethod
    def gen_sqrtm(cls):
        def track_sqrtm(self, dtype, n, blocksize):
            n = int(n)
            dtype = np.dtype(dtype)
            blocksize = int(blocksize)
            A = np.random.rand(n, n)
            if dtype == np.complex128:
                A = A + 1j*np.random.rand(n, n)
            return measure('scipy.linalg.sqrtm(A, disp=False, blocksize=blocksize)')

        for dtype in ('float64', 'complex128'):
            for n in (64, 256):
                for blocksize in sorted(list(set([n, 32, 64]))):
                    yield track_sqrtm, dtype, str(n), str(blocksize)
