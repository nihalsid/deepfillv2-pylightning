import numpy as np


class NormalizeRange(object):

    def __init__(self, minval, maxval, all_data_min=0, all_data_max=1):
        self.minval = minval
        self.maxval = maxval
        self.all_data_min = all_data_min
        self.all_data_max = all_data_max

    def __call__(self, t):
        normalized = self.minval + (self.maxval - self.minval) * (t - self.all_data_min)/(self.all_data_max - self.all_data_min)
        return normalized


class ToNumpyRGB256(object):

    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, t):
        return (((np.transpose(t, axes=[1, 2, 0]) - self.minval) / (self.maxval - self.minval)) * 255.0).astype(np.uint8)
