from __future__ import division, print_function, absolute_import
from .common import Benchmark, measure

from math import sqrt

# Import testing parameters
from scipy.optimize._tstutils import (methods, mstrings, functions,
     fstrings, description)


class Zeros(Benchmark):
    group_by = {
        'gen_all': ['row', 'col'],
    }

    @classmethod
    def gen_all(self):
        a = .5
        b = sqrt(3)
        repeat = 2000

        def time(self, func_str, meth_str):
            func = functions[fstrings.index(func_str.replace('_','.'))]
            meth = methods[mstrings.index(meth_str.replace('_','.'))]
            meth(func, a, b)

        time.goal_time = 0.5

        for func_str in fstrings:
            for meth_str in mstrings:
                yield time, func_str.replace('.','_'), meth_str.replace('.','_')
