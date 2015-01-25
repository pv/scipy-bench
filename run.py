#!/usr/bin/env python
"""
run.py COMMAND

"""
from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import subprocess


EXTRA_PATH = ['/usr/lib/ccache', '/usr/lib/f90cache',
              '/usr/local/lib/ccache', '/usr/local/lib/f90cache']


def main():
    sys.exit(run_asv(*sys.argv[1:]))


def run_asv(*args):
    cmd = ['asv'] + list(args)
    cwd = os.path.abspath(os.path.dirname(__file__))
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join(EXTRA_PATH + env.get('PATH', '').split(os.pathsep))
    return subprocess.call(cmd, env=env, cwd=cwd)


if __name__ == "__main__":
    main()
