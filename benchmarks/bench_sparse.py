"""
Simple benchmarks for the sparse module
"""
from __future__ import division, print_function, absolute_import

import warnings
import time

import numpy
import numpy as np
from numpy import ones, array, asarray, empty, random, zeros

from scipy import sparse
from scipy.sparse import (csr_matrix, coo_matrix, dia_matrix, lil_matrix,
                          dok_matrix, rand, SparseEfficiencyWarning)

from .common import Benchmark, SimpleTimer


import scipy
import sys


def random_sparse(m,n,nnz_per_row):
    rows = numpy.arange(m).repeat(nnz_per_row)
    cols = numpy.random.random_integers(low=0,high=n-1,size=nnz_per_row*m)
    vals = numpy.random.random_sample(m*nnz_per_row)
    return coo_matrix((vals,(rows,cols)),(m,n)).tocsr()


# TODO move this to a matrix gallery and add unittests
def poisson2d(N,dtype='d',format=None):
    """
    Return a sparse matrix for the 2D Poisson problem
    with standard 5-point finite difference stencil on a
    square N-by-N grid.
    """
    if N == 1:
        diags = asarray([[4]],dtype=dtype)
        return dia_matrix((diags,[0]), shape=(1,1)).asformat(format)

    offsets = array([0,-N,N,-1,1])

    diags = empty((5,N**2),dtype=dtype)

    diags[0] = 4  # main diagonal
    diags[1:] = -1  # all offdiagonals

    diags[3,N-1::N] = 0  # first lower diagonal
    diags[4,N::N] = 0  # first upper diagonal

    return dia_matrix((diags,offsets),shape=(N**2,N**2)).asformat(format)


class Arithmetic(Benchmark):
    def setup(self):
        self.matrices = {}
        # matrices.append( ('A','Identity', sparse.eye(500**2,format='csr')) )
        self.matrices['A'] = poisson2d(250,format='csr')
        self.matrices['B'] = poisson2d(250,format='csr')**2

    @classmethod
    def gen_arithmetic(cls):
        def track(self, format, X, Y, op):
            vars = dict([(var, mat.asformat(format)) 
                         for (var, mat) in self.matrices.items()])
            x,y = vars[X],vars[Y]
            fn = getattr(x,op)
            fn(y)  # warmup

            t = SimpleTimer()
            for _ in t:
                fn(y)
            return t.timing

        for format in ['csr']:
            for X,Y in [('A','A'),('A','B'),('B','A'),('B','B')]:
                for op in ['__add__','__sub__','multiply','__mul__']:
                    yield track, format, X, Y, op


class Sort(Benchmark):
    @classmethod
    def gen_sort(cls):
        """sort CSR column indices"""

        matrices = []
        matrices.append(('Rand10', '1e4', '10'))
        matrices.append(('Rand25', '1e4', '25'))
        matrices.append(('Rand50', '1e4', '50'))
        matrices.append(('Rand100', '1e4', '100'))
        matrices.append(('Rand200', '1e4', '200'))

        def track(self, name, N, K):
            N = int(float(N))
            K = int(float(K))
            A = random_sparse(N,N,K)

            t = SimpleTimer()
            for _ in t:
                A.has_sorted_indices = False
                A.indices[:2] = 2,1
                A.sort_indices()
            return t.timing

        for name, N, K in matrices:
            yield track, name, N, K


class Matvec(Benchmark):
    @classmethod
    def _get_matrices(cls):
        matrices = {}

        matrices['Identity_dia'] = sparse.eye(10**4,format='dia')
        matrices['Identity_csr'] = sparse.eye(10**4,format='csr')
        matrices['Poisson5pt_lil'] = poisson2d(300,format='lil')
        matrices['Poisson5pt_dok'] = poisson2d(300,format='dok')
        matrices['Poisson5pt_dia'] = poisson2d(300,format='dia')
        matrices['Poisson5pt_coo'] = poisson2d(300,format='coo')
        matrices['Poisson5pt_csr'] = poisson2d(300,format='csr')
        matrices['Poisson5pt_csc'] = poisson2d(300,format='csc')
        matrices['Poisson5pt_bsr'] = poisson2d(300,format='bsr')

        A = sparse.kron(poisson2d(150),ones((2,2))).tobsr(blocksize=(2,2))
        matrices['Block2x2_csr'] = A.tocsr()
        matrices['Block2x2_bsr'] = A

        A = sparse.kron(poisson2d(100),ones((3,3))).tobsr(blocksize=(3,3))
        matrices['Block3x3_csr'] = A.tocsr()
        matrices['Block3x3_bsr'] = A
        return matrices

    def setup(self):
        self.matrices = self.__class__._get_matrices()
        self.x = ones(max(A.shape[1] for A in self.matrices.values()), 
                      dtype=float)

    @classmethod
    def gen_matvec(cls):
        def time(self, name):
            A = self.matrices[name]
            x = self.x[:A.shape[1]]
            y = A * x
        for name in cls._get_matrices().keys():
            yield time, name


class Matvecs(Benchmark):
    def setup(self):
        self.matrices = {}
        self.matrices['dia'] = poisson2d(300,format='dia')
        self.matrices['coo'] = poisson2d(300,format='coo')
        self.matrices['csr'] = poisson2d(300,format='csr')
        self.matrices['csc'] = poisson2d(300,format='csc')
        self.matrices['bsr'] = poisson2d(300,format='bsr')
        A = self.matrices['dia']
        self.x = ones((A.shape[1], 10), dtype=A.dtype)

    @classmethod
    def gen_matvecs(cls):
        def time(self, fmt):
            A = self.matrices[fmt]
            y = A*self.x

        for fmt in ['dia', 'coo', 'csr', 'csc', 'bsr']:
            yield time, fmt


class Matmul(Benchmark):
    def setup(self):
        H1, W1 = 1, 100000
        H2, W2 = W1, 1000
        C1 = 10
        C2 = 1000000

        random.seed(0)

        matrix1 = lil_matrix(zeros((H1, W1)))
        matrix2 = lil_matrix(zeros((H2, W2)))
        for i in range(C1):
            matrix1[random.randint(H1), random.randint(W1)] = random.rand()
        for i in range(C2):
            matrix2[random.randint(H2), random.randint(W2)] = random.rand()
        self.matrix1 = matrix1.tocsr()
        self.matrix2 = matrix2.tocsr()

    def time_large(self):
        for i in range(100):
            matrix3 = self.matrix1 * self.matrix2


class Construction(Benchmark):
    def setup(self):
        self.matrices = {}
        self.matrices['Empty'] = csr_matrix((10000,10000))
        self.matrices['Identity'] = sparse.eye(10000)
        self.matrices['Poisson5pt'] = poisson2d(100)
        self.formats = {'lil': lil_matrix, 'dok': dok_matrix}

    @classmethod
    def gen_construction(cls):
        """build matrices by inserting single values"""

        def track(self, name, format):
            A = self.matrices[name]
            cls = self.formats[format]

            A = A.tocoo()

            t = SimpleTimer()
            for _ in t:
                T = cls(A.shape)
                for i,j,v in zip(A.row,A.col,A.data):
                    T[i,j] = v
            return t.timing

        for mat in ['Empty', 'Identity', 'Poisson5pt']:
            for format in ['lil', 'dok']:
                yield track, mat, format


class Conversion(Benchmark):
    def setup(self):
        self.A = poisson2d(100)

    @classmethod
    def gen_conversion(cls):
        formats = ['csr','csc','coo','dia','lil','dok']

        def track(self, fromfmt, tofmt):
            A = self.A
            base = getattr(A,'to' + fromfmt)()

            result = np.nan
            try:
                fn = getattr(base,'to' + tofmt)
            except:
                pass
            else:
                x = fn()  # warmup
                t = SimpleTimer()
                for _ in t:
                    x = fn()
                return t.timing

            return result

        for fromfmt in formats:
            for tofmt in formats:
                yield track, fromfmt, tofmt


class Getset(Benchmark):
    def setup(self):
        self.A = rand(1000, 1000, density=1e-5)

    @classmethod
    def _getset_bench(cls, kernel_name, kernel, formats):
        def track(self, N, spat, fmt):
            A = self.A
            N = int(N)
            spat = (spat != 'False')

            # indices to assign to
            i, j = [], []
            while len(i) < N:
                n = N - len(i)
                ip = numpy.random.randint(0, A.shape[0], size=n)
                jp = numpy.random.randint(0, A.shape[1], size=n)
                i = numpy.r_[i, ip]
                j = numpy.r_[j, jp]
            v = numpy.random.rand(n)

            if N == 1:
                i = int(i)
                j = int(j)
                v = float(v)

            if fmt == 'dok' and N > 500:
                return np.nan

            base = A.asformat(fmt)

            m = base.copy()
            if spat:
                kernel(m, i, j, v)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', SparseEfficiencyWarning)

                iter = 0
                total_time = 0
                while total_time < 0.2 and iter < 5000:
                    if not spat:
                        m = base.copy()
                    a = time.clock()
                    kernel(m, i, j, v)
                    total_time += time.clock() - a
                    iter += 1

            result = total_time/float(iter)
            return result

        track.__name__ = "track_" + kernel_name

        for N in [1, 10, 100, 1000, 10000]:
            for spat in [False, True]:
                for fmt in formats:
                    yield track, str(N), str(spat), fmt

    @classmethod
    def gen_setitem(cls):
        def kernel(A, i, j, v):
            A[i, j] = v

        for v in cls._getset_bench("fancy_setitem", kernel, 
                                   ['csr', 'csc', 'lil', 'dok']):
            yield v[0], v[1], v[2], v[3]

    @classmethod
    def gen_getitem(cls):
        def kernel(A, i, j, v=None):
            A[i, j]
        for v in cls._getset_bench("fancy_getitem", kernel,
                                   ['csr', 'csc', 'lil']):
            yield v[0], v[1], v[2], v[3]
