..  -*- rst -*-

===========
scipy-bench
===========

Benchmarking Scipy with Airspeed Velocity.

Usage
-----

Run a benchmark against currently installed Scipy version (don't
record the result)::

    asv dev --bench bench_sparse.Arithmetic

Run a benchmark using Scipy ``runtests.py`` (don't record results)::

    cd ../scipy
    python runtests.py -g --python ../scipy-bench/run.sh dev --bench bench_sparse.Arithmetic

Run benchmarks against various Scipy git versions (record results)::

    asv run --bench bench_sparse.Arithmetic

More in `ASV documentation`_

.. _ASV documentation: https://spacetelescope.github.io/asv/


Writing benchmarks
------------------

The basics in `ASV documentation`_ apply. We structure things here as
follows::

    # benchmarks/bench_<submodule>.py
    from .common import Benchmark, run_monitored
    from scipy.foo import bar

    class SomeFeature(Benchmark):
        def setup(self):
            # setup code for all benchmarks in the class
            self.data = {'a': 'asd', 'b': 'blashyrkh'}

        def time_bar(self):
            # ASV provides the timing code itself! using this is
            # preferred
            bar()

        def track_bar_memory_usage(self):
            # Track an arbitrary performance number. 
            # Best not used for timing tests
            code = """
            from scipy.foo import bar
            bar()
            """
            time, peak_mem = run_monitored(code)
            return peak_mem

        @classmethod
        def gen_bar(cls):
            # This can be used to easily generate many tests where only
            # some parameter(s) vary. The special status of the function
            # is recognized from that its name starts with gen_

            # The name of the following function should conform to ASV
            # naming, and start with time_ or track_. It becomes a class
            # method via lesser magic in the Benchmark class. The name of
            # the benchmark is derived by appending "_{arg1}_{arg2}"
            # to the name of the function.
            def time_foo(self, arg1, arg2): 
                data1 = self.data[arg1] 
                data2 = self.data[arg2]
                foo(data1, data2)

            # Loop over arguments. All arguments must be strings that can
            # appear in function names!
            for arg in ['a', 'b']:
                    yield time_foo, arg

The ``gen_`` methods in general should only contain two things:
(i) the function returned, (ii) for loops over a string list literals.

Avoid putting anything that requires much memory into the ``gen_``
methods.  Rather, put the allocations in the ``setup`` method, or do
them inside the function yielded from ``gen_``.  Avoid writing
``gen_`` methods that can fail (eg for old Scipy versions).  It's OK
to fail in the setup() method and in the function yielded from
``gen_``.

Preparing arrays should generally be put in the ``setup`` method
rather than the ``time_`` methods, to avoid counting preparation time
together with the time of the benchmarked operation.

I don't recommend using ASV's ``mem_`` benchmarks, since tracking the
size of some object is probably not very useful for Scipy --- tracking
peak memory is likely more useful.
