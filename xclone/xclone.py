# xclone - Statistical modelling of copy number variations in single cells
# Author: Rongting Huang
# Date: 2022-09-25

import os
import sys
import time
import subprocess
import numpy as np
import multiprocessing
from scipy.io import mmread
from optparse import OptionParser, OptionGroup

from .version import __version__

START_TIME = time.time()


def show_progress(RV=None):
    return RV

def main():
    import warnings
    warnings.filterwarnings('error')

    # parse command line options
    parser = OptionParser()


if __name__ == "__main__":
    main()
