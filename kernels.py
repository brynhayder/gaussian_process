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
# TODO: Make descriptor that is appropriate for updating params. Maybe simple property good enough.
# TODO: Use hypothesis to write some tests for these
# Note that some of these kernels only make sense for specific domains


class KernelMixIn(object):
    @staticmethod
    def _squeeze_arrays(*arrays):
        """To make args compatible with np.ufunc.outer"""
        return tuple(x.squeeze() for x in arrays)


class BrownianBridge(KernelMixIn):
    """Covariance kernel for Brownian Bridge process"""
    def __call__(self, x, y):
        x, y = self._squeeze_arrays(x, y)
        return np.subtract(np.minimum.outer(x, y),
                           np.multiply.outer(x, y))


class Wiener(KernelMixIn):
    """Covariance kernel for Wiener Process"""
    def __call__(self, x, y):
        x, y = self._squeeze_arrays(x, y)
        return np.minimum.outer(x, y)


class RadialBasisFunction(KernelMixIn):
    def __init__(self, length_scale=1.):
        self.length_scale = length_scale

    def __call__(self, x, y):
        x, y = self._squeeze_arrays(x, y)
        dist = np.subtract.outer(x, y)
        return np.exp(- np.power(dist, 2.) / (2. * self.length_scale ** 2.))


class OrnsteinUhlenbeck(KernelMixIn):
    def __init__(self, length_scale=1.):
        self.length_scale = length_scale

    def __call__(self, x, y):
        x, y = self._squeeze_arrays(x, y)
        dist = np.subtract.outer(x, y)
        return np.exp(- np.abs(dist) / self.length_scale)


class RationalQuadratic(KernelMixIn):
    """K(s, t) = (1 + (s - t)^2) ^ - alpha"""
    def __init__(self, alpha=1., length_scale=1.):
        self.alpha = alpha
        self.length_scale = length_scale

    def __call__(self, x, y):
        x, y = self._squeeze_arrays(x, y)
        dist = np.subtract.outer(x, y)
        c = 2 * self.alpha * self.length_scale ** 2
        return np.power(1 + np.power(dist, 2)/c, -1 * self.alpha)


class Periodic(KernelMixIn):
    def __init__(self, length_scale=1., periodicity=1.):
        self.length_scale = length_scale
        self.periodicity = periodicity

    def __call__(self, x, y):
        x, y = self._squeeze_arrays(x, y)
        dist = np.subtract.outer(x, y)
        return np.exp(- 2. * np.sin(np.pi * dist / self.periodicity) ** 2 / self.length_scale ** 2)


class Constant(KernelMixIn):
    """K(s, t) = sigma"""
    def __init__(self, sigma=1.):
        self.sigma = sigma

    def __call__(self, x, y):
        return np.full((x.shape[0], y.shape[0]), self.sigma)


class WhiteNoise(KernelMixIn):
    """K(s, t) = sigma**2 I"""
    def __init__(self, sigma=1.):
        self.sigma = sigma

    def __call__(self, x, y):
        # This is clearly not an optimal implementation
        cov = np.zeros((x.shape[0], y.shape[0]))
        np.fill_diagonal(cov, self.sigma ** 2)
        return cov
