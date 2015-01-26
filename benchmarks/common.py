"""
Airspeed Velocity benchmark utilities
"""
from __future__ import division, absolute_import, print_function

import sys
import re
import time
import textwrap
import tempfile
import subprocess
import inspect
import collections
import itertools

from six import with_metaclass


ASV_SPECIAL_ATTRIBUTES = ['unit', 'setup', 'timeout', 'goal_time', 'number',
                          'repeat', 'timer']


class BenchmarkMetaclass(type):
    """
    Autogenerate additional methods to a benchmark class:

    1. Treat all gen_* methods as generators, which are assumed to
       yield ``(func,) + tuple(args)``. For each yielded value, a test method
       that calls func with the given arguments is added.

    2. Add numpy benchmark suite compatible wrapper functions for all
       ASV-style time_*, mem_*, track_* methods

    """

    def __new__(mcls, cls_name, bases, dct):
        cls = super(BenchmarkMetaclass, mcls).__new__(mcls, cls_name, bases, dct)

        benchmark_info = {}

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

        # NumpyTest compatibility
        for name, obj in list(dct.items()):
            if name.startswith('time_') or name.startswith('mem_') or name.startswith('track_'):
                func = lambda self, name=name: self._do_bench(name)
                func.__name__ = "bench_" + name
                add_to_dict(func.__name__, func)

        # Generator support
        for name, obj in list(dct.items()):
            if name.startswith('gen_'):
                obj = getattr(cls, name)
                if not inspect.isgeneratorfunction(obj):
                    raise ValueError("%s.%s must be a generator function" % (
                        cls_name, name,))

                # Insert ASV benchmark routines
                names = []
                basename = ""
                for r in obj():
                    basename = r[0].__name__
                    if any(not isinstance(x, str) for x in r[1:]):
                        raise ValueError("One of the arguments yielded from %s.%s is not a string" % (
                            cls_name, name))
                    bench_name = basename + "_" + "_".join(x for x in r[1:])
                    if not re.match(r'^[a-zA-Z0-9_]+$', bench_name):
                        raise ValueError("Benchmark name %r is not a valid Python function name" % (
                            bench_name))
                    if not (bench_name.startswith('time_') or
                            bench_name.startswith('track_') or
                            bench_name.startswith('mem_')):
                        raise ValueError("Benchmark function names must be ASV compatible and "
                                         "start with time_ OR mem_ OR track_")
                    func = lambda self, f=r[0], a=r[1:]: f(self, *a)
                    func.__name__ = bench_name
                    for attr in ['__doc__'] + ASV_SPECIAL_ATTRIBUTES:
                        if hasattr(r[0], attr):
                            setattr(func, attr, getattr(r[0], attr))
                    add_to_dict(bench_name, func)
                    benchmark_info[bench_name] = r[1:]
                    names.append(bench_name)

                # Insert NumpyTest style benchmarks
                bench_name = "bench_" + name[4:]
                group_by = getattr(cls, 'group_by', {}).get(name)
                func = lambda self, groupby=group_by, basename=bench_name, names=names: self._do_bench_multi(basename, names,
                                                                                                             group_by)
                func.__name__ = bench_name
                add_to_dict(bench_name, func)

        # Store some information for pretty-printing results in Numpy
        # style benchmarks
        if hasattr(cls, 'benchmark_info'):
            cls.benchmark_info.update(benchmark_info)
        else:
            cls.benchmark_info = benchmark_info

        return cls


class Benchmark(with_metaclass(BenchmarkMetaclass, object)):
    # The methods in this class are **only** needed for NumpyTest
    # benchmark support
    __test__ = True

    def _do_bench_multi(self, basename, names, group_by):
        name = self.__class__.__name__ + "." + basename
        sys.stdout.write("\n\n")
        sys.stdout.write(name)
        sys.stdout.write("\n" + "="*len(name) + "\n")

        if group_by is None:
            for name in names:
                self._do_bench(name, quiet=False)
        else:
            results = []
            for name in names:
                info = self.benchmark_info.get(name)
                result = self._do_bench(name)
                results.append(info + (result,))
            sys.stdout.write('\n\n')

            self._print_result_table(sys.stdout, results, group_by)

    def _do_bench(self, method_name, quiet=False):
        from asv.benchmark import TimeBenchmark, MemBenchmark, TrackBenchmark
        if method_name.startswith('time_'):
            bm_type = TimeBenchmark
        elif method_name.startswith('mem_'):
            bm_type = MemBenchmark
        elif method_name.startswith('track_'):
            bm_type = TrackBenchmark
        else:
            raise ValueError("Unknown benchmark type prefix in method name %r" % (method_name,))

        info = self.benchmark_info.get(method_name)

        benchmark = bm_type.from_class_method(self.__class__, method_name)
        benchmark.do_setup()
        try:
            result = benchmark.do_run()
            try:
                result = float(result)
            except (ValueError, TypeError):
                raise ValueError("Benchmark functions should return "
                                 "a floating point number describing "
                                 "the result (in seconds or bytes)")
        finally:
            benchmark.do_teardown()

        if not quiet:
            if info:
                args = [str(x) for x in info]
                text = " | ".join([str(x).center(15) for x in args] + [str(result).center(15)])
            else:
                text = repr(result)

            sys.stdout.write(text)
            sys.stdout.write("\n")
            sys.stdout.flush()

        return result


    def _print_result_table(self, stream, results, group_by):
        def sort_key(r):
            k = []
            for x in r:
                try:
                    k.append(float(x))
                except ValueError:
                    k.append(x)
            return k

        def unique(values):
            value_set = set()
            result = []
            for value in values:
                if value not in value_set:
                    value_set.add(value)
                    result.append(value)
            return result

        results.sort(key=sort_key)
        possible_values = [unique([r[k] for r in results])
                           for k in range(len(results[0]) - 1)]

        col_values = [v for status, v in zip(group_by, possible_values)
                      if status == 'col']
        row_values = [v for status, v in zip(group_by, possible_values)
                      if status == 'row']

        stream.write("| ")
        stream.write(" | ".join(''.center(15) for value in row_values))
        stream.write("| ")
        stream.write(" | ".join(", ".join(map(str, values)).center(15)
                                    for values in itertools.product(*col_values)))
        last_row_value = ()
        for r in results:
            row_value = [value for status, value in zip(group_by, r) 
                         if status == 'row']
            if row_value != last_row_value:
                stream.write("\n")
                vals = []
                for status, value in zip(group_by, r):
                    if status == 'row':
                        vals.append(str(value).center(15))
                stream.write("| ")
                stream.write(" | ".join(vals))
            stream.write("| %s " % (str(r[-1]).center(15),))
            last_row_value = row_value
        stream.write("\n")
        stream.flush()


class SimpleTimer(object):
    def __init__(self, duration=0.5, min_iter=1):
        self.start = None
        self.initial = None
        self.duration = duration
        self.numiter = 0
        self.min_time = 1e99
        self.min_iter = min_iter

    def __iter__(self):
        self.initial = time.clock()
        return self

    def next(self):
        if self.start is not None:
            t = time.clock()
            d = t - self.start
            if d < self.min_time:
                self.min_time = d
            if t > self.initial + self.duration:
                if self.numiter >= self.min_iter:
                    raise StopIteration()
            self.numiter += 1
        self.start = time.clock()
        return None

    __next__ = next

    @property
    def timing(self):
        return self.min_time


def measure(code_str):
    """
    Return elapsed time for executing code in the namespace of the caller.
    """
    frame = sys._getframe(1)
    locs, globs = frame.f_locals, frame.f_globals

    code = compile(code_str, '<string>', 'exec')
    t = SimpleTimer()
    for _ in t:
        exec(code, globs, locs)
    return t.timing


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


def get_mem_info():
    """Get information about available memory"""
    if not sys.platform.startswith('linux'):
        raise RuntimeError("Memory information implemented only for Linux")

    info = {}
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            p = line.split()
            info[p[0].strip(':').lower()] = float(p[1]) * 1e3
    return info


def set_mem_rlimit(max_mem=None):
    """
    Set address space rlimit
    """
    import resource
    if max_mem is None:
        mem_info = get_mem_info()
        max_mem = int(mem_info['memtotal'] * 0.7)
    cur_limit = resource.getrlimit(resource.RLIMIT_AS)
    if cur_limit[0] > 0:
        max_mem = min(max_mem, cur_limit[0])

    resource.setrlimit(resource.RLIMIT_AS, (max_mem, cur_limit[1]))
