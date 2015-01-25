""" Benchmark functions for fftpack.pseudo_diffs module
"""
from __future__ import division, absolute_import, print_function
from .common import Benchmark, measure

import sys

from numpy import arange, sin, cos, pi, exp, tanh, sign
from scipy.fftpack import diff, fft, ifft, tilbert, hilbert, shift, fftfreq


def random(size):
    return rand(*size)


def direct_diff(x,k=1,period=None):
    fx = fft(x)
    n = len(fx)
    if period is None:
        period = 2*pi
    w = fftfreq(n)*2j*pi/period*n
    if k < 0:
        w = 1 / w**k
        w[0] = 0.0
    else:
        w = w**k
    if n > 2000:
        w[250:n-250] = 0.0
    return ifft(w*fx).real


def direct_tilbert(x,h=1,period=None):
    fx = fft(x)
    n = len(fx)
    if period is None:
        period = 2*pi
    w = fftfreq(n)*h*2*pi/period*n
    w[0] = 1
    w = 1j/tanh(w)
    w[0] = 0j
    return ifft(w*fx)


def direct_hilbert(x):
    fx = fft(x)
    n = len(fx)
    w = fftfreq(n)*n
    w = 1j*sign(w)
    return ifft(w*fx)


def direct_shift(x,a,period=None):
    n = len(x)
    if period is None:
        k = fftfreq(n)*1j*n
    else:
        k = fftfreq(n)*2j*pi/period*n
    return ifft(fft(x)*exp(k*a)).real


class Bench(Benchmark):
    group_by = {
        'gen_diff': ['row', 'col'],
        'gen_tilbert': ['row', 'col'],
        'gen_hilbert': ['row', 'col'],
        'gen_shift': ['row', 'col'],
    }

    @classmethod
    def _gen_random(cls, name, call, direct_call):
        def track(self, size, direct):
            size = int(size)

            x = arange(size)*2*pi/size
            a = 1
            if size < 2000:
                f = sin(x)*cos(4*x)+exp(sin(3*x))
                sf = sin(x+a)*cos(4*(x+a))+exp(sin(3*(x+a)))
            else:
                f = sin(x)*cos(4*x)
                sf = sin(x+a)*cos(4*(x+a))

            if direct == 'direct':
                return measure(direct_call)
            else:
                return measure(call)

        track.__name__ = "track_" + name
        track.unit = "s"

        for size,repeat in [(100,1500),(1000,300),
                            (256,1500),
                            (512,1000),
                            (1024,500),
                            (2048,200),
                            (2048*2,100),
                            (2048*4,50),
                            ]:
            yield track, str(size), "direct"
            yield track, str(size), "fft"

    @classmethod
    def gen_diff(cls):
        for func, size, direct in cls._gen_random("diff",
                                                  "diff(f,3)",
                                                  "direct_diff(f,3)"):
            yield func, size, direct

    @classmethod
    def gen_tilbert(cls):
        for func, size, direct in cls._gen_random("tilbert",
                                                  "tilbert(f,1)",
                                                  "direct_tilbert(f,1)"):
            yield func, size, direct

    @classmethod
    def gen_hilbert(cls):
        for func, size, direct in cls._gen_random("hilbert",
                                                  "hilbert(f)",
                                                  "direct_hilbert(f)"):
            yield func, size, direct

    @classmethod
    def gen_shift(cls):
        for func, size, direct in cls._gen_random("shift",
                                                  "shift(f,a)",
                                                  "direct_shift(f,a)"):
            yield func, size, direct
