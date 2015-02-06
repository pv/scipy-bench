#!/usr/bin/env python
"""
Convenience wrapper around the ``asv`` command; just sets environment
variables and chdirs to the correct place.
"""
from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import subprocess
import sysconfig


EXTRA_PATH = ['/usr/lib/ccache', '/usr/lib/f90cache',
              '/usr/local/lib/ccache', '/usr/local/lib/f90cache']


def main():
    sys.exit(run_asv(*sys.argv[1:]))


def run_asv(*args):
    cmd = ['asv'] + list(args)
    cwd = os.path.abspath(os.path.dirname(__file__))
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join(EXTRA_PATH + env.get('PATH', '').split(os.pathsep))

    # Required to make ccache work properly --- pip builds and installs stuff
    # to paths with random components in directory names, which confuses ccache.
    #
    # To work around this, we need to encourage ccache to be sloppy, and to
    # remove the -g flag from compiler flags --- it causes file names to be
    # embedded to the object files, which ccache does not like.
    #env['CCACHE_SLOPPINESS'] = 'file_macro,time_macros'
    #env['CCACHE_UNIFY'] = '1'
    #env['CFLAGS'] = drop_g_flag(sysconfig.get_config_var('CFLAGS'))
    #env['OPT'] = drop_g_flag(sysconfig.get_config_var('OPT'))
    #env['LDSHARED'] = drop_g_flag(sysconfig.get_config_var('LDSHARED'))

    env['ATLAS'] = 'None'
    return subprocess.call(cmd, env=env, cwd=cwd)


def drop_g_flag(flags):
    """
    Drop -g from command line flags
    """
    if not flags:
        return flags
    return " ".join(x for x in flags.split() if x not in ('-g', '-g1', '-g2', '-g3'))


if __name__ == "__main__":
    main()
