import os

from data import common
from data import mstestsrdata

import numpy as np

import torch
import torch.utils.data as data

class MSBenchmark(mstestsrdata.MSSRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(MSBenchmark, self).__init__(args, name=name)

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('', '.png')

