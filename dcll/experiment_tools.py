#!/usr/bin/env python
# -----------------------------------------------------------------------------
# File Name : experiment_tools.py
# Author: Emre Neftci
#
# Creation Date : Thu 26 Jul 2018 09:13:34 AM PDT
# Last Modified :
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : Apache License, Version 2.0
# -----------------------------------------------------------------------------

import os
import numpy as np


def annotate(d, text='', filename='notes.txt'):
    """Create a file FILENAME in the directory D with contents TEXT."""
    f = open(os.path.join(d, filename), 'w')
    f.write(text)
    f.close()


def save_source(directory):
    """Save all the Python scripts from the current directory into the results directory."""
    import tarfile
    import glob
    h = tarfile.open(directory + '/exp_scripts.tar.bz2', 'w:bz2')
    all_src = []
    all_src += glob.glob('*.py')
    for i in all_src:
        h.add(i)
    h.close()


def mksavedir(pre='results/'):
    """
    Creates a results directory in the subdirectory `pre`.
    The directory name is given by ###__dd_mm_yy, where ### is the next unused 3-digit number.
    """
    import time
    import fnmatch

    if not pre.endswith('/'):
        pre += '/'
    if not os.path.exists(pre):
        os.makedirs(pre)
    prelist = np.sort(fnmatch.filter(os.listdir(pre), '[0-9][0-9][0-9]__*'))

    if len(prelist) == 0:
        expDirN = '001'
    else:
        expDirN = '%03d' % (
            int((prelist[len(prelist) - 1].split('__'))[0]) + 1)

    directory = os.path.join(pre, expDirN + '__' + '%d-%m-%Y')
    directory = time.strftime(directory, time.localtime())
    assert not os.path.exists(directory)

    os.mkdir(directory)
    directory += '/'

    fh = open(directory + time.strftime('%H:%M:%S', time.localtime()), 'w')
    fh.close()

    print(('Created experiment directory {0}'.format(directory)))
    return directory
