from __future__ import division, absolute_import, print_function
from .common import Benchmark, measure

import sys
import numpy.linalg as nl
import scipy.linalg as sl

from numpy.testing import assert_
from numpy.random import rand


def random(size):
    return rand(*size)


class Bench(Benchmark):
    group_by = {
        'gen_solve': ['row', 'col', 'col'],
        'gen_inv': ['row', 'col', 'col'],
        'gen_det': ['row', 'col', 'col'],
    }

    @classmethod
    def _gen_generic(cls, name, call):
        def track(self, size, contig, numpy_str):
            numpy_f = getattr(nl, name)
            scipy_f = getattr(sl, name)
            size = int(size)

            a = random([size,size])
            # larger diagonal ensures non-singularity:
            for i in range(size):
                a[i,i] = 10*(.1+a[i,i])
            b = random([size])

            if contig != 'contig':
                a = a[-1::-1,-1::-1]  # turn into a non-contiguous array
                assert_(not a.flags['CONTIGUOUS'])

            return measure(numpy_str + "_" + call)

        track.__name__ = "track_" + name
        track.unit = "s"

        for size,repeat in [(20,1000),(100,150),(500,2),(1000,1)][:-1]:
            for contig in ['contig', 'nocont']:
                for numpy_str in ['numpy', 'scipy']:
                    yield track, str(size), contig, numpy_str

    @classmethod
    def gen_solve(cls):
        for func, a, b, c in cls._gen_generic('solve', 'f(a, b)'):
            yield func, a, b, c

    @classmethod
    def gen_inv(cls):
        for func, a, b, c in cls._gen_generic('inv', 'f(a)'):
            yield func, a, b, c

    @classmethod
    def gen_det(cls):
        for func, a, b, c in cls._gen_generic('det', 'f(a)'):
            yield func, a, b, c

    @classmethod
    def gen_eigvals(cls):
        for func, a, b, c in cls._gen_generic('eigvals', 'f(a)'):
            yield func, a, b, c

    @classmethod
    def gen_svd(cls):
        for func, a, b, c in cls._gen_generic('svd', 'f(a)'):
            yield func, a, b, c
