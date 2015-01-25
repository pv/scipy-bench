"""
Make Airspeed Velocity benchmarks Numpy benchmark suite compatible
"""
from __future__ import division, absolute_import, print_function

import sys
import re
import time
import textwrap
import tempfile
import subprocess
import inspect

from six import with_metaclass


class BenchmarkMetaclass(type):
    """
    Autogenerate additional methods to a benchmark class:

    1. Treat all gen_* methods as generators, which are assumed to
       yield ``(func,) + tuple(args)``. For each yielded value, a test method
       that calls func with the given arguments is added.

    2. Add numpy benchmark suite compatible wrapper functions for all
       ASV-style time_*, mem_*, track_* methods

    """

    def __init__(cls, cls_name, bases, dct):
        super(BenchmarkMetaclass, cls).__init__(cls_name, bases, dct)

        def add_to_dict(key, value):
            if hasattr(cls, key):
                raise ValueError(("Entry %r already exists in benchmark class %r; "
                                  "remove your custom definition") % (
                    key, cls_name,))
            setattr(cls, key, value)

        # Enforce ASV style
        for name, obj in list(dct.items()):
            if name.startswith('bench_') or name.startswith('test_'):
                raise ValueError("%s.%s starts with bench_ or test_, which is not allowed; "
                                 "test should be written to be ASV compatible" % (cls_name, name))

        # Generator support
        for name, obj in list(dct.items()):
            if name.startswith('gen_'):
                obj = getattr(cls, name)
                if not inspect.isgeneratorfunction(obj):
                    raise ValueError("%s.%s must be a generator function" % (
                        cls_name, name,))

                del dct[name]

                # Insert ASV benchmark routines
                names = []
                for r in obj():
                    bench_name = r[0].__name__ + "_" + "_".join(x for x in r[1:])
                    if not (bench_name.startswith('time_') or
                            bench_name.startswith('track_') or
                            bench_name.startswith('mem_')):
                        raise ValueError("Benchmark function names must be ASV compatible and "
                                         "start with time_ OR mem_ OR track_")
                    func = lambda self, f=r[0], a=r[1:]: f(self, *a)
                    func.__name__ = bench_name
                    add_to_dict(bench_name, func)
                    names.append(bench_name)


class Benchmark(with_metaclass(BenchmarkMetaclass, object)):
    pass


class SimpleTimer(object):
    def __init__(self, duration=0.5):
        self.start = None
        self.end = None
        self.duration = duration
        self.numiter = 0

    def __iter__(self):
        self.start = time.clock()
        return self

    def next(self):
        if time.clock() > self.start + self.duration:
            if self.numiter > 0:
                self.end = time.clock()
                raise StopIteration()
        self.numiter += 1
        return None

    __next__ = next

    @property
    def timing(self):
        return (self.end - self.start) / self.numiter


def run_monitored(code):
    """
    Run code in a new Python process, and monitor peak memory usage.

    Returns
    -------
    duration : float
        Duration in seconds (including Python startup time)
    peak_memusage : float
        Peak memory usage (rough estimate only) in bytes

    """
    if not sys.platform.startswith('linux'):
        raise RuntimeError("Peak memory monitoring only works on Linux")

    code = textwrap.dedent(code)
    process = subprocess.Popen([sys.executable, '-c', code])

    peak_memusage = -1

    start = time.time()
    while True:
        ret = process.poll()
        if ret is not None:
            break

        with open('/proc/%d/status' % process.pid, 'r') as f:
            procdata = f.read()

        m = re.search('VmRSS:\s*(\d+)\s*kB', procdata, re.S | re.I)
        if m is not None:
            memusage = float(m.group(1)) * 1e3
            peak_memusage = max(memusage, peak_memusage)

        time.sleep(0.01)

    process.wait()

    duration = time.time() - start

    if process.returncode != 0:
        raise AssertionError("Running failed:\n%s" % code)

    return duration, peak_memusage
