#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import os
import re

import cv2
import numpy as np
import core.transforms as transforms
import torch.utils.data

from glob import glob

import pickle as pkl
# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

class DataSet(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, data_path, scale_list):
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        self._data_path, self._scale_list = data_path, scale_list
        self._construct_db()

    def _construct_db(self):
        """Constructs the db."""
        self._db = []

        for im_path in glob(f'{self._data_path}/*'):
            for yi in [0, 0.5, 1.0]:
                for xi in [0, 0.5, 1.0]:
                     self._db.append({"im_path": im_path, 'xi': xi, 'yi': yi})

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = transforms.color_norm(im, _MEAN, _SD)
        return im

    def __getitem__(self, index):
        # Load the image
        try:
            ori_im = cv2.imread(self._db[index]["im_path"])
            xi = self._db[index]['xi']
            yi = self._db[index]['yi']
            im_list = []

            x, y, _ = ori_im.shape
            c_width = int(x/2)
            c_height = int(y/2)

            im = ori_im[int(xi*c_width): int((xi+1)*c_width), int(yi*c_height): int((yi+1)*c_height)]

            for scale in self._scale_list:
                if scale == 1.0:
                    im_np = im.astype(np.float32, copy=False)
                    im_list.append(im_np)
                elif scale < 1.0:
                    im_resize = cv2.resize(im, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    im_np = im_resize.astype(np.float32, copy=False)
                    im_list.append(im_np)
                elif scale > 1.0:
                    im_resize = cv2.resize(im, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    im_np = im_resize.astype(np.float32, copy=False)
                    im_list.append(im_np)      
                else:
                    assert()

        except Exception as e:
            print('error: ', self._db[index]["im_path"])
            print(e)

        for idx in range(len(im_list)):
            im_list[idx] = self._prepare_im(im_list[idx])
        return im_list

    def __len__(self):
        return len(self._db)
