#!/usr/bin/env python
"""
Some basic kernels for Gaussian Processes.

Kernels must accept two (n, 1) arrays and return a matrix.

Notes:
------
    - Some of these should just be functions, not classes. I made them all classes to unify the API.
"""
__author__ = 'brynhayder'

# TODO: Use hypothesis to write some tests for these

# Note that some of these kernels only make sense for specific domains
# multiplying by numbers?

import numpy as np


class _KernelBase:
    def get_params(self):
        raise NotImplementedError("Must implement get_params")

    def n_params(self):
        return len(self.get_params())

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(["{}={:.2f}".format(k, v) for k, v in self.get_params().items()])
        )

    def copy_with(self, *params):
        """
        Make a new version of this kernel with new parameters

        Args:
            *params: Parameters for the __init__ of the kernel class.

        Returns:
            (self.__class__): The new kernel.
        """
        return self.__class__(*params)

    @staticmethod
    def _squeeze_arrays(*arrays):
        """To make args compatible with np.ufunc.outer"""
        return tuple(x.squeeze() for x in arrays)

    def __mul__(self, other):
        if not isinstance(other, _KernelBase):
            raise NotImplementedError("Multiplication only defined between kernels")
        return ProductKernel(lhs=self, rhs=other)

    def __add__(self, other):
        if not isinstance(other, _KernelBase):
            raise NotImplementedError("Addition only defined between kernels")
        return SumKernel(lhs=self, rhs=other)


class _BinaryOperationMixIn(_KernelBase):
    _operator = None

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def n_params(self):
        return self.lhs.n_params() + self.rhs.n_params()

    def __repr__(self):
        return "({} {} {})".format(
            repr(self.lhs),
            self._operator,
            repr(self.rhs)
        )

    def get_params(self):
        return {
            "lhs": self.lhs.get_params(),
            "rhs": self.rhs.get_params()
        }

    def copy_with(self, *params):
        n = self.lhs.n_params()
        p1, p2 = params[:n], params[n:]
        return self.__class__(
            lhs=self.lhs.copy_with(*p1),
            rhs=self.rhs.copy_with(*p2)
        )


class ProductKernel(_BinaryOperationMixIn):
    _operator = "x"

    def __call__(self, x, y):
        return self.lhs(x, y) * self.rhs(x, y)


class SumKernel(_BinaryOperationMixIn):
    _operator = "+"

    def __call__(self, x, y):
        return self.lhs(x, y) + self.rhs(x, y)


class RadialBasisFunction(_KernelBase):
    def __init__(self, length_scale=1.):
        self.length_scale = length_scale

    def get_params(self):
        return dict(
            length_scale=self.length_scale
        )

    def __call__(self, x, y):
        x, y = self._squeeze_arrays(x, y)
        dist = np.subtract.outer(x, y)
        return np.exp(- np.power(dist, 2.) / (2. * self.length_scale ** 2.))


class RationalQuadratic(_KernelBase):
    """K(s, t) = (1 + (s - t)^2) ^ - alpha"""
    def __init__(self, alpha=1., length_scale=1.):
        self.alpha = alpha
        self.length_scale = length_scale

    def get_params(self):
        return dict(
            alpha=self.alpha,
            length_scale=self.length_scale
        )

    def __call__(self, x, y):
        x, y = self._squeeze_arrays(x, y)
        dist = np.subtract.outer(x, y)
        c = 2 * self.alpha * self.length_scale ** 2
        return np.power(1 + np.power(dist, 2)/c, - self.alpha)


class Periodic(_KernelBase):
    def __init__(self, length_scale=1., periodicity=1.):
        self.length_scale = length_scale
        self.periodicity = periodicity

    def get_params(self):
        return dict(
            length_scale=self.length_scale,
            periodicity=self.periodicity
        )

    def __call__(self, x, y):
        x, y = self._squeeze_arrays(x, y)
        dist = np.subtract.outer(x, y)
        return np.exp(
            - 2. * np.sin(np.pi * dist / self.periodicity) ** 2 / self.length_scale ** 2
        )


class Constant(_KernelBase):
    """K(s, t) = const"""
    def __init__(self, const=1.):
        self.const = const

    def get_params(self):
        return dict(
            const=self.const
        )

    def __call__(self, x, y):
        return np.full((x.shape[0], y.shape[0]), self.const)


class WhiteNoise(_KernelBase):
    """K(s, t) = sigma**2 I"""
    def __init__(self, sigma=1.):
        self.sigma = sigma

    def get_params(self):
        return dict(
            sigma=self.sigma
        )

    def __call__(self, x, y):
        # This is clearly not an optimal implementation memory wise
        cov = np.zeros((x.shape[0], y.shape[0]))
        np.fill_diagonal(cov, self.sigma ** 2)
        return cov


class OrnsteinUhlenbeck(_KernelBase):
    def __init__(self, length_scale=1.):
        self.length_scale = length_scale

    def get_params(self):
        return dict(
            length_scale=self.length_scale
        )

    def __call__(self, x, y):
        x, y = self._squeeze_arrays(x, y)
        dist = np.subtract.outer(x, y)
        return np.exp(- np.abs(dist) / self.length_scale)


class BrownianBridge(_KernelBase):
    """Covariance kernel for Brownian Bridge process"""
    def get_params(self):
        return {}

    def __call__(self, x, y):
        x, y = self._squeeze_arrays(x, y)
        return np.subtract(np.minimum.outer(x, y),
                           np.multiply.outer(x, y))


class Wiener(_KernelBase):
    """Covariance kernel for Wiener Process"""
    def get_params(self):
        return {}

    def __call__(self, x, y):
        x, y = self._squeeze_arrays(x, y)
        return np.minimum.outer(x, y)
