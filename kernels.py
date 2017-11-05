#!/usr/bin/env python
"""
Some basic kernels for Gaussian Processes.

Kernels must accept two (n, 1) arrays and return a matrix.

Notes:
------
    - Some of these should just be functions, not classes. I made them all classes to unify the API.
"""
import numpy as np

__author__ = 'brynhayder'

# TODO: Implement some kernel arithmetic
# TODO: Some of these kernels only make sense for specific domains

class KernelMixIn(object):
    def _prep_args(self, x, y):
        return x.squeeze(), y.squeeze()


class BrownianBridge(KernelMixIn):
    """Covariance kernel for Brownian Bridge process"""
    def __call__(self, x, y):
        x, y = self._prep_args(x, y)
        return np.subtract(np.minimum.outer(x, y),
                           np.multiply.outer(x, y))


class Wiener(KernelMixIn):
    """Covariance kernel for Wiener Process"""
    def __call__(self, x, y):
        x, y = self._prep_args(x, y)
        return np.minimum.outer(x, y)


class RadialBasisFunction(KernelMixIn):
    def __init__(self, length_scale=1., sigma=1.):
        self.length_scale = length_scale
        self.sigma = sigma

    def __call__(self, x, y):
        x, y = self._prep_args(x, y)
        dist = np.subtract.outer(x, y)
        return self.sigma ** 2. * np.exp(- np.power(dist, 2.)
                                         / (2. * self.length_scale ** 2.))


