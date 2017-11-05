#!/usr/bin/env python
"""
Toy implementation of Gaussian Process for 1-d Regression.
Only for educational purposes, use something like scikit-learn for production use.

Hopefully pretty obvious to see how everything works.
"""
import numpy as np

__author__ = 'brynhayder'
__all__ = ['GaussianProcess']

# TODO: Finish Plotter class
# TODO: Write tests and examples
# TODO: implement for more than 1-d Regression (?)
# TODO: Deal with non-zero prior mean (?)


class GaussianProcess(object):
    """ Basic implementation of Gaussian Process for 1-d Regression """
    def __init__(self, kernel, random_state=None):
        """
        Gaussian Process Regression with prior mean of 0

        In the below:
        - arr is any np.array of floats and shape (n, 1)
        - mat is any np.array floats and shape (n, n)

        Args:
            kernel (callable(arr, arr) -> mat): The prior specification of the covariance kernel.
            random_state (np.random.RandomState): Optional. If None will make own and use seed 0.

        Notes:
            - You need to do .fit before you can take any posterior samples or make any predictions.
        """
        self.kernel = kernel
        self.random_state = random_state if random_state is not None else np.random.RandomState(seed=0)
        # set by .fit
        self._is_fit = False
        self._train_x = None
        self._train_y = None
        self._k = None
        self._k_inv = None
        self._noise_level = None

    def _sample(self, mean, cov, size):
        """
        Return samples from multivariate normal using internal RandomState.

        See np.random.multivariate_normal.
        """
        return (self.random_state
                    .multivariate_normal(mean=mean,
                                         cov=cov,
                                         size=size,
                                         check_valid='warn')
                    .T)

    @property
    def plot(self):
        """ Expose simple plotting API """
        raise NotImplementedError('Plotter not yet implemented')
        # return GPRPlotter(process=self)

    def sample_prior(self, x, size=1, return_std=False):
        """
        Evaluate samples from the prior distribution
        at a grid of points `x`

        Args:
            x (np.array): 1-d array of floats.
            size: int, default 1. number of samples you want.
            return_std (bool): Optional, default False. Return std at each point.

        Returns:
            np.array, shape (n, size). If return_std == False (default).
            (np.array, shape (n, size), np.array, shape (n,)). If return_std == True.
        """
        # This conversion is a bit of an inefficiency, since the kernel
        # will do .squeeze() anyway. I think worth it for a consistent API
        # in this toy example.
        x = np.atleast_2d(x)
        cov = self.kernel(x, x)
        samples = self._sample(mean=np.zeros(x.shape[0]),
                               cov=cov,
                               size=size)
        return (samples, np.sqrt(np.diag(cov))) if return_std else samples

    def fit(self, x, y, noise_level=0.):
        """
        Fit the process to some training examples (`x`, `y`).

        We are fitting the model
        y = f(x) + e, where e ~ N(0, noise_level**2)
        so we are able to incorporate noise in the observation of y.

        Args:
            x (np.array): array-like, shape (n, 1), floats.
            y (np.array): array-like, shape (n, 1), floats.
            noise_level (float, np.array): Optional. Noise in observation of `y` values.
                Can be float or array of shape (x.shape[0],).
        """
        x, y = np.atleast_2d(x, y)

        self._train_x = x
        self._train_y = y
        self._noise_level = noise_level
        k = self.kernel(self._train_x, self._train_x)
        self._k = k + noise_level ** 2 * np.eye(k.shape[0])
        # noinspection PyTypeChecker
        self._k_inv = np.linalg.inv(self._k)
        self._is_fit = True
        return None

    def posterior_moments(self, x):
        """
        Calculate posterior mean and covariance of process,
        evaluated at `x` (conditioned on training data).

        Args:
            x (np.array): array-like, shape (n, 1), floats.

        Returns:
            (np.array (n, 1), np.array (n, n))
            posterior mean, posterior covariance evaluated at x.

        Raises:
            ValueError if you've not yet fit the process.
        """
        if not self._is_fit:
            raise ValueError('Need to fit the process to some training data first!')
        x = np.atleast_2d(x)
        k_star = self.kernel(x, self._train_x)
        k_star_star = self.kernel(x, x)
        posterior_mean = np.dot(np.dot(k_star, self._k_inv), self._train_y)
        posterior_cov = k_star_star - np.dot(k_star, np.dot(self._k_inv, k_star.T))
        return posterior_mean, posterior_cov

    def sample_posterior(self, x, size=1, reg=1e-12):
        """
        Draw samples from posterior process evaluated at `x`.
        (Conditioned on training data.)

        Args:
            x (np.array): array-like, shape (n, 1), floats.
            size (int): Optional, default 1. Number of samples.
            reg (float): Add `reg` amplitude white noise to diagonal of cov
                before sampling to ensure it is positive-definite.

        Returns:
            np.array, floats, (`x`.shape[0], size)

        Raises:
            ValueError if you've not yet fit the process.
        """
        if not self._is_fit:
            raise ValueError('Need to fit the process to some training data first!')
        posterior_mean, posterior_cov = self.posterior_moments(x)
        # add small amplitude noise to diagonal of cov before sampling
        cov = posterior_cov + reg * np.diag(np.random.randn(posterior_cov.shape[0]))
        return self._sample(mean=posterior_mean.squeeze(),
                            cov=cov,
                            size=size)

    def predict(self, x, return_std=False):
        """
        Make predictions at `x`. Evaluates the posterior mean of
        the process, conditional on the training data.

        Args:
            x (np.array): array-like, shape (n, 1), floats.
            return_std (bool): Optional, default False. Return std at each point.

        Returns:
            np.array, shape (n, 1). The posterior mean only, if return_std == False (default).
            (np.array, shape (n, 1), np.array, shape (n)). The posterior mean with
            standard error, if return_std == True.

        Raises:
            ValueError if you've not yet fit the process.
        """
        if not self._is_fit:
            raise ValueError('Need to fit the process to some training data first!')
        posterior_mean, posterior_cov = self.posterior_moments(x)
        return (posterior_mean, np.sqrt(np.diag(posterior_cov))) if return_std else posterior_mean


# BE: This is currently super simple, but it would be good to add error bars, etc.
class GPRPlotter(object):
    """ Simple plotting methods for 1-d Gaussian Process Regression """
    def __init__(self, process):
        self.process = process

    def __call__(self, plot, *args, **kwargs):
        """
        Just an accessor method really. self(`plot`, ***) just does
        self.`plot`(***).

        Args:
            plot (string): name of the method you want to call.
            *args: args for method.
            **kwargs: kwargs for method.

        Returns:
            (fig, ax) # All plotting methods return this.

        Raises:
            AttributeError. When you ask for a plotting method that doesn't exist.
        """
        try:
            plotter = getattr(self, plot)
        except AttributeError:
            raise AttributeError('No plotting method named {}.'.format(plot))
        else:
            return plotter(*args, **kwargs)

    def prediction(self, grid, ax=None):
        """
        Plot prediction of process at `grid`, conditional on training data.

        Args:
            grid: Points on which to evaluate the process.
            ax (matplotlib ax): Obvs.

        Returns:

        """
        pass

    def posterior(self, ax=None, n_samples=5):
        """

        Args:
            ax:
            n_samples:

        Returns:

        """
        pass

    def prior(self, ax=None, n_samples=5):
        """

        Args:
            ax:
            n_samples:

        Returns:

        """
        pass