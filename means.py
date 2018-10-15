#!/usr/bin/env python
"""
--------------------------------
project: labs
created: 12/10/2018 16:30
---------------------------------

"""

import numpy as np


# class Constant


class _MeanBase:
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


class ConstantMean(_MeanBase):
    def __init__(self, const):
        self.const = const

    def get_params(self):
        return dict(const=self.const)

    def __call__(self, x):
        return np.full(x.shape, self.const)

