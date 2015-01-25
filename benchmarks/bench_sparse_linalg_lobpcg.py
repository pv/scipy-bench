from __future__ import division, absolute_import, print_function
from .common import Benchmark, measure

from functools import partial

import numpy as np
from numpy.testing import assert_allclose

import time
from scipy import *
from scipy.linalg import eigh, orth, cho_factor, cho_solve
import scipy.sparse
from scipy.sparse.linalg import lobpcg
from scipy.sparse.linalg.interface import LinearOperator


def _sakurai(n):
    """ Example taken from
        T. Sakurai, H. Tadano, Y. Inadomi and U. Nagashima
        A moment-based method for large-scale generalized eigenvalue problems
        Appl. Num. Anal. Comp. Math. Vol. 1 No. 2 (2004) """

    A = scipy.sparse.eye(n, n)
    d0 = array(r_[5,6*ones(n-2),5])
    d1 = -4*ones(n)
    d2 = ones(n)
    B = scipy.sparse.spdiags([d2,d1,d0,d1,d2],[-2,-1,0,1,2],n,n)

    k = arange(1,n+1)
    w_ex = sort(1./(16.*pow(cos(0.5*k*pi/(n+1)),4)))  # exact eigenvalues

    return A, B, w_ex


def _mikota_pair(n):
    # Mikota pair acts as a nice test since the eigenvalues
    # are the squares of the integers n, n=1,2,...
    x = arange(1,n+1)
    B = diag(1./x)
    y = arange(n-1,0,-1)
    z = arange(2*n-1,0,-2)
    A = diag(z)-diag(y,-1)-diag(y,1)
    return A.astype(float), B.astype(float)


def _as2d(ar):
    if ar.ndim == 2:
        return ar
    else:  # Assume 1!
        aux = nm.array(ar, copy=False)
        aux.shape = (ar.shape[0], 1)
        return aux


def _precond(LorU, lower, x):
    y = cho_solve((LorU, lower), x)
    return _as2d(y)


class Bench(Benchmark):
    group_by = {
        'gen_mikota': ['row', 'col'],
        'gen_sakurai': ['row', 'col']
    }

    @classmethod
    def gen_mikota(cls):
        def track_mikota(self, n, soltype):
            n = int(n)
            m = 10
            shape = (n, n)
            A, B = _mikota_pair(n)
            desired_evs = np.square(np.arange(1, m+1))

            if soltype == 'lobpcg':
                tt = time.clock()
                X = rand(n, m)
                X = orth(X)
                LorU, lower = cho_factor(A, lower=0, overwrite_a=0)
                M = LinearOperator(shape,
                        matvec=partial(_precond, LorU, lower),
                        matmat=partial(_precond, LorU, lower))
                eigs, vecs = lobpcg(A, X, B, M, tol=1e-4, maxiter=40)
                eigs = sorted(eigs)
                elapsed = time.clock() - tt
                assert_allclose(eigs, desired_evs)
            else:
                tt = time.clock()
                w = eigh(A, B, eigvals_only=True, eigvals=(0, m-1))
                elapsed = time.clock() - tt
                assert_allclose(w, desired_evs)
            return elapsed

        for n in 128, 256, 512, 1024, 2048:
            for soltype in ['lobpcg', 'eigh']:
                yield track_mikota, str(n), soltype

    @classmethod
    def gen_sakurai(cls):
        def track_sakurai(self, n, soltype):
            n = int(n)
            m = 3

            shape = (n, n)
            A, B, all_eigenvalues = _sakurai(n)
            desired_evs = all_eigenvalues[:m]

            if soltype == 'lobpcg':
                tt = time.clock()
                X = rand(n, m)
                eigs, vecs, resnh = lobpcg(A, X, B, tol=1e-6, maxiter=500,
                        retResidualNormsHistory=1)
                w_lobpcg = sorted(eigs)
                elapsed = time.clock() - tt
                assert_allclose(w_lobpcg, desired_evs, 1e-7, 1e-5)
            else:
                tt = time.clock()
                A_dense = A.A
                B_dense = B.A
                w_eigh = eigh(A_dense, B_dense, eigvals_only=True, eigvals=(0, m-1))
                elapsed = time.clock() - tt
                assert_allclose(w_eigh, desired_evs, 1e-7, 1e-5)
            return elapsed

        for n in 50, 400, 2400:
            for soltype in ['lobpcg', 'eigh']:
                yield track_sakurai, str(n), soltype
