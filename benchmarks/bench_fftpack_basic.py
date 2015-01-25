""" Test functions for fftpack.basic module
"""
from __future__ import division, absolute_import, print_function
from .common import Benchmark, measure

import sys
from scipy.fftpack import ifft, fft, fftn, irfft, rfft

from numpy.testing import assert_array_almost_equal

from numpy import arange, asarray, zeros, dot, exp, pi, double, cdouble
import numpy.fft

from numpy.random import rand


def random(size):
    return rand(*size)


def direct_dft(x):
    x = asarray(x)
    n = len(x)
    y = zeros(n,dtype=cdouble)
    w = -arange(n)*(2j*pi/n)
    for i in range(n):
        y[i] = dot(exp(i*w),x)
    return y


def direct_idft(x):
    x = asarray(x)
    n = len(x)
    y = zeros(n,dtype=cdouble)
    w = arange(n)*(2j*pi/n)
    for i in range(n):
        y[i] = dot(exp(i*w),x)/n
    return y


class Fft(Benchmark):
    group_by = {
        'gen_fft': ['row', 'row', 'col', 'col']
    }

    @classmethod
    def gen_fft(cls):
        def track_random(self, func, size, cmplx, numpy_str):
            from numpy.fft import fft as numpy_fft
            from numpy.fft import ifft as numpy_ifft
            from numpy.fft import rfft as numpy_rfft
            from numpy.fft import irfft as numpy_irfft

            size = int(size)

            if cmplx == 'cmplx':
                x = random([size]).astype(cdouble)+random([size]).astype(cdouble)*1j
            else:
                x = random([size]).astype(double)

            if size > 500:
                y = fft(x)
            else:
                y = direct_dft(x)
            assert_array_almost_equal(fft(x),y)
            if numpy_str == 'numpy':
                return measure('numpy_' + func + '(x)')
            else:
                return measure(func + '(x)')

        track_random.unit = "s"

        for func in ['fft', 'ifft', 'rfft', 'irfft']:
            for size,repeat in [(100,7000),(1000,2000),
                                (256,10000),
                                (512,10000),
                                (1024,1000),
                                (2048,1000),
                                (2048*2,500),
                                (2048*4,500),
                                ]:
                for cmplx in ['real', 'cmplx']:
                    if func in ('rfft', 'irfft') and cmplx != 'real':
                        continue
                    for numpy_str in ['scipy', 'numpy']:
                        yield track_random, func, str(size), cmplx, numpy_str


class Fftn(Benchmark):
    group_by = {
        'gen_random': ['row', 'col', 'col']
    }

    @classmethod
    def gen_random(self):
        def track_random(self, size, cmplx, numpy_str):
            from numpy.fft import fftn as numpy_fftn

            size = map(int, size.split("x"))

            if cmplx != 'cmplx':
                x = random(size).astype(double)
            else:
                x = random(size).astype(cdouble)+random(size).astype(cdouble)*1j

            func = 'fftn'
            if numpy_str == 'numpy':
                return measure('numpy_' + func + '(x)')
            else:
                return measure(func + '(x)')

        track_random.unit = "s"

        for size,repeat in [("100x100",100),("1000x100",7),
                            ("256x256",10),
                            ("512x512",3),
                            ]:
                for cmplx in ['real', 'cmplx']:
                    for numpy_str in ['scipy', 'numpy']:
                        yield track_random, str(size), cmplx, numpy_str
