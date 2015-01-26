"""benchmarks for the scipy.sparse.linalg._expm_multiply module"""
from __future__ import division, print_function, absolute_import
from .common import Benchmark

import time
import math

import numpy as np
from numpy.testing import assert_allclose

import scipy.linalg
from scipy.sparse.linalg import expm_multiply


def random_sparse_csr(m, n, nnz_per_row):
    # Copied from the scipy.sparse benchmark.
    rows = np.arange(m).repeat(nnz_per_row)
    cols = np.random.random_integers(low=0, high=n-1, size=nnz_per_row*m)
    vals = np.random.random_sample(m*nnz_per_row)
    M = scipy.sparse.coo_matrix((vals,(rows,cols)), (m,n), dtype=float)
    return M.tocsr()


def random_sparse_csc(m, n, nnz_per_row):
    # Copied from the scipy.sparse benchmark.
    rows = np.arange(m).repeat(nnz_per_row)
    cols = np.random.random_integers(low=0, high=n-1, size=nnz_per_row*m)
    vals = np.random.random_sample(m*nnz_per_row)
    M = scipy.sparse.coo_matrix((vals,(rows,cols)), (m,n), dtype=float)
    # Use csc instead of csr, because sparse LU decomposition
    # raises a warning when I use csr.
    return M.tocsc()


class MultiBenchmark(Benchmark):
    """
    Lazy man's way of splitting multiple test results obtained from a
    single function to multiple ASV test results.  Stuff has to be
    stored in the class, since ASV uses one instance per test.
    """
    def run_if_needed(self):
        if not hasattr(self.__class__, '_results'):
            self.run()

    def store_result(self, name, value):
        if name not in self.__class__.result_names:
            raise ValueError("Result name %r not in __class__.result_names" % (name,))
        if not hasattr(self.__class__, '_results'):
            self.__class__._results = {}
        self._results[name] = value

    def get_result(self, name):
        return self.__class__._results[name]


class ExpmMultiply(MultiBenchmark):
    result_names = ['full', 'sparse']

    @classmethod
    def gen_all(cls):
        def track(self, name):
            self.run_if_needed()
            return self.get_result(name)
        for name in cls.result_names:
            yield track, name

    def _help_bench_expm_multiply(self, A, i, j):
        n = A.shape[0]
        tm_start = time.clock()
        A_dense = A.toarray()
        tm_end = time.clock()

        # computing full expm of the dense array...
        tm_start = time.clock()
        A_expm = scipy.linalg.expm(A_dense)
        full_expm_entry = A_expm[i, j]
        tm_end = time.clock()
        print('expm(A)[%d, %d]:' % (i, j), full_expm_entry)
        self.store_result('full', tm_end - tm_start)


        # computing only column', j, 'of expm of the sparse matrix...
        tm_start = time.clock()
        v = np.zeros(n, dtype=float)
        v[j] = 1
        A_expm_col_j = expm_multiply(A, v)
        expm_col_entry = A_expm_col_j[i]
        tm_end = time.clock()
        print('expm(A)[%d, %d]:' % (i, j), expm_col_entry)
        self.store_result('sparse', tm_end - tm_start)

        assert np.allclose(full_expm_entry, expm_col_entry)

    def run(self):
        np.random.seed(1234)
        n = 2000
        i = 100
        j = 200
        shape = (n, n)
        nnz_per_row = 25
        tm_start = time.clock()
        A = random_sparse_csr(n, n, nnz_per_row)
        tm_end = time.clock()
        self._help_bench_expm_multiply(A, i, j)


class Expm(MultiBenchmark):
    result_names = ['dense_30', 
                    'dense_100', 
                    'dense_300',
                    'sparse_30',
                    'sparse_100',
                    'sparse_300']

    @classmethod
    def gen_all(cls):
        def track(self, name):
            self.run_if_needed()
            return self.get_result(name)
        for name in cls.result_names:
            yield track, name

    def run(self):
        # check three roughly exponentially increasing matrix orders
        np.random.seed(1234)
        for n in (30, 100, 300):
            # Let the number of nonzero entries per row
            # scale like the log of the order of the matrix.
            nnz_per_row = int(math.ceil(math.log(n)))
            shape = (n, n)

            # time the sampling of a random sparse matrix
            tm_start = time.clock()
            A_sparse = random_sparse_csc(n, n, nnz_per_row)
            tm_end = time.clock()
            tm_sampling = tm_end - tm_start

            # first format conversion
            tm_start = time.clock()
            A_dense = A_sparse.toarray()
            tm_end = time.clock()
            tm_first_fmt = tm_end - tm_start

            # sparse matrix exponential
            tm_start = time.clock()
            A_sparse_expm = scipy.linalg.expm(A_sparse)
            tm_end = time.clock()
            tm_sparse = tm_end - tm_start

            # dense matrix exponential
            tm_start = time.clock()
            A_dense_expm = scipy.linalg.expm(A_dense)
            tm_end = time.clock()
            tm_dense = tm_end - tm_start

            # second format conversion
            tm_start = time.clock()
            A_sparse_expm_as_dense = A_sparse_expm.toarray()
            tm_end = time.clock()
            tm_second_fmt = tm_end - tm_start

            # sum the format conversion times
            tm_fmt = tm_first_fmt + tm_second_fmt

            # check that the matrix exponentials are the same
            assert_allclose(A_sparse_expm_as_dense, A_dense_expm)

            # write the rows
            self.store_result('sparse_%d' % (n,), tm_sparse)
            self.store_result('dense_%d' % (n,), tm_dense)
        print()


