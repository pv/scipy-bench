from __future__ import division, absolute_import, print_function
from .common import Benchmark, set_mem_rlimit, run_monitored, get_mem_info, SimpleTimer

import os
import sys
import re
import subprocess
import time
import textwrap
import tempfile
import collections
from io import BytesIO

import numpy as np
from scipy.io import savemat, loadmat


class MemUsage(Benchmark):
    @classmethod
    def _get_sizes(cls):
        sizes = collections.OrderedDict([
            ('1M', 1e6),
            ('10M', 10e6),
            ('100M', 100e6),
            ('300M', 300e6),
            ('500M', 500e6),
            ('1000M', 1000e6),
        ])
        return sizes

    def setup(self):
        set_mem_rlimit()
        self.sizes = self.__class__._get_sizes()

    @classmethod
    def gen_loadmat(cls):
        def track_loadmat(self, size, compressed):
            size = int(self.sizes[size])
            compressed = (compressed == 'compress')

            mem_info = get_mem_info()
            max_size = int(mem_info['memtotal'] * 0.7)//4

            if size > max_size:
                return np.nan

            # Setup temp file, make it fit in memory
            f = tempfile.NamedTemporaryFile(suffix='.mat')
            os.unlink(f.name)

            try:
                x = np.random.rand(size//8).view(dtype=np.uint8)
                savemat(f.name, dict(x=x), do_compression=compressed, oned_as='row')
                del x
            except MemoryError:
                return np.nan

            code = """
            from scipy.io import loadmat
            loadmat('%s')
            """ % (f.name,)
            time, peak_mem = run_monitored(code)

            return peak_mem / size

        track_loadmat.unit = "ratio = peak memory / theoretical minimum requirement"

        for size in cls._get_sizes().keys():
            for compressed in ('nocompress', 'compress'):
                yield track_loadmat, size, str(compressed)

    @classmethod
    def gen_savemat(cls):
        def track_savemat(self, size, compressed):
            size = int(self.sizes[size])
            compressed = (compressed == 'compress')

            mem_info = get_mem_info()
            max_size = int(mem_info['memtotal'] * 0.7)//4

            if size > max_size:
                return np.nan

            # Setup temp file, make it fit in memory
            f = tempfile.NamedTemporaryFile(suffix='.mat')
            os.unlink(f.name)

            code = """
            import numpy as np
            from scipy.io import savemat
            x = np.random.rand(%d//8).view(dtype=np.uint8)
            savemat('%s', dict(x=x), do_compression=%r, oned_as='row')
            """ % (size, f.name, compressed)
            time, peak_mem = run_monitored(code)
            return peak_mem / size

        track_savemat.unit = "ratio = peak memory / theoretical minimum requirement"

        for size in cls._get_sizes().keys():
            for compressed in ('nocompress', 'compress'):
                yield track_savemat, size, compressed


class StructArr(Benchmark):
    @staticmethod
    def make_structarr(n_vars, n_fields, n_structs):
        var_dict = {}
        for vno in range(n_vars):
            vname = 'var%00d' % vno
            end_dtype = [('f%d' % d, 'i4', 10) for d in range(n_fields)]
            s_arrs = np.zeros((n_structs,), dtype=end_dtype)
            var_dict[vname] = s_arrs
        return var_dict

    @classmethod
    def gen_savemat(cls):
        def track_savemat(self, n_vars, n_fields, n_structs, compression):
            n_vars = int(n_vars)
            n_fields = int(n_fields)
            n_structs = int(n_structs)
            compression = (compression == 'compress')

            var_dict = StructArr.make_structarr(n_vars, n_fields, n_structs)
            str_io = BytesIO()

            t = SimpleTimer()
            for _ in t:
                savemat(str_io, var_dict, do_compression=compression)
            return t.timing

        track_savemat.unit = "s"

        for n_vars, n_fields, n_structs in ((10, 10, 20), (20, 20, 40),
                                            (30, 30, 50)):
            for compression in ('nocompress', 'compress'):
                yield track_savemat, str(n_vars), str(n_fields), str(n_structs), compression

    @classmethod
    def gen_loadmat(cls):
        def track_loadmat(self, n_vars, n_fields, n_structs, compression):
            n_vars = int(n_vars)
            n_fields = int(n_fields)
            n_structs = int(n_structs)
            compression = (compression == 'compress')

            var_dict = StructArr.make_structarr(n_vars, n_fields, n_structs)
            str_io = BytesIO()

            savemat(str_io, var_dict, do_compression=compression)
            t = SimpleTimer()
            for _ in t:
                loadmat(str_io)
            return t.timing

        track_loadmat.unit = "s"

        for n_vars, n_fields, n_structs in ((10, 10, 20), (20, 20, 40),
                                            (30, 30, 50)):
            for compression in ('nocompress', 'compress'):
                yield track_loadmat, str(n_vars), str(n_fields), str(n_structs), compression
