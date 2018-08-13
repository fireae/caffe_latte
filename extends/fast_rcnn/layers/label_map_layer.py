# encoding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import caffe
import numpy as np
from utils.config import cfg

class LabelMapLayer(caffe.Layer):

    def setup(self, bottom, top):
        height, width = bottom[0].data.shape[-2:]
        top[0].reshape(1,1, height, width)
        top[1].reshape(1,1, height, width)
        top[2].reshape(1,1, height, width)
        top[3].reshape(1,1, height, width)

    def forward(self, bottom, top):
        #params
        batch_size = 32
        fg_fraction = 1.0
        
