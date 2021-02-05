import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data

class MSSRData(data.Dataset):
    def __init__(self, args, name=''):
        self.args = args
        self.name = name
        self.scale = args.test_scale
        self.benchmark = True
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)

        list_hr, list_lr = self._scan()
        self.images_hr, self.images_lr = list_hr, list_lr

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{:.2f}/{}{}'.format(
                        s, filename, self.ext[1]
                    )
                ))

        return names_hr, names_lr

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = imageio.imread(f_hr)
        lr = imageio.imread(f_lr)

        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        ih, iw = lr.shape[:2]
        hr = hr[0:int(round(ih*scale)), 0:int(round(iw*scale))]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
