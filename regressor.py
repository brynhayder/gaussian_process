#!/usr/bin/env python
__author__ = 'brynhayder'
__all__ = ['GaussianProcessRegressor']


import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize



def _zero_prior_mean(x):
    return np.zeros(x.shape[0])


class GaussianProcessRegressor(object):
    """ Basic implementation of Gaussian Process for 1-d Regression """

    def __init__(self, kernel, random_state=None, noise_level=0.):
        """
        Gaussian Process Regression with prior mean of 0

        In the below:
        - arr is any np.array of floats and shape (n, 1)
        - mat is any np.array floats and shape (n, n)

        Args:
            kernel (callable(arr, arr) -> mat): The prior specification of the covariance kernel.
            random_state (np.random.RandomState): Optional. If None will make own and use seed 0.
            noise_level (float, np.array): Optional. Fixed noise in observation of `y` values.
                Can be float or array of shape (x.shape[0],).

        Notes:
            - You need to do .fit before you can take any posterior samples or make any predictions.
        """
        self.prior_mean = _zero_prior_mean
        self.kernel = kernel
        self.random_state = random_state if random_state is not None else np.random.RandomState(seed=0)
        self.noise_level = noise_level
        # set by .fit
        self._is_fit = False
        self.train_x = None
        self.train_y = None
        self._k = None
        self._k_cholesky_lt = None

    def __repr__(self):
        return """
{}(
    kernel={},
    random_state={},
    noise_level={}
)
""".format(
            self.__class__.__name__,
            repr(self.kernel),
            repr(self.random_state),
            self.noise_level
        )

    def _sample(self, mean, cov, size):
        """
        Return samples from multivariate normal using internal RandomState.

        See np.random.multivariate_normal.
        """
        return (
            self.random_state
                .multivariate_normal(
                    mean=mean,
                    cov=cov,
                    size=size,
                    check_valid='warn'
            ).transpose()
        )

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
        x = np.atleast_2d(x)
        cov = self.kernel(x, x)
        samples = self._sample(
                mean=self.prior_mean(x),
                cov=cov,
                size=size
        )
        return (samples, np.sqrt(np.diag(cov))) if return_std else samples

    def fit(self, x, y):
        """
        Fit the process to some training examples (`x`, `y`).

        We are fitting the model
        y = f(x) + e, where e ~ N(0, noise_level**2)
        so we are able to incorporate noise in the observation of y.

        Args:
            x (np.array): array-like, shape (n, 1), floats.
            y (np.array): array-like, shape (n, 1), floats.

        Returns:
            self (GaussianProcessRegressor): The fitted model.
        """
        x, y = np.atleast_2d(x, y)

        self.train_x = x
        self.train_y = y
        k = self.kernel(self.train_x, self.train_x)
        self._k = k + self.noise_level ** 2 * np.eye(k.shape[0])
        self._k_cholesky_lt = np.linalg.cholesky(self._k)
        self._is_fit = True
        return self

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
            raise ValueError('Need to fit before calculating posterior moments.')
        x = np.atleast_2d(x)
        k_star = self.kernel(x, self.train_x)
        k_star_star = self.kernel(x, x)
        posterior_mean = np.dot(
            k_star,
            cho_solve((self._k_cholesky_lt, True), self.train_y)
        )
        posterior_cov = k_star_star - np.dot(
            k_star,
            cho_solve((self._k_cholesky_lt, True), k_star.T)
        )
        return posterior_mean, posterior_cov

    def sample_posterior(self, x, size=1, return_std=False, jitter=0.):
        """
        Draw samples from posterior process evaluated at `x`.
        (Conditioned on training data.)

        Args:
            x (np.array): array-like, shape (n, 1), floats.
            size (int): Optional, default 1. Number of samples.
            return_std (bool): Optional, default False. Return std at each point.
            jitter (float): Add `reg` to diagonal of cov before sampling to ensure
            it is positive-definite.

        Returns:
            np.array, floats, (`x`.shape[0], size)

        Raises:
            ValueError if you've not yet fit the process.
        """
        if not self._is_fit:
            raise ValueError('Need to fit the process to some training data first!')
        posterior_mean, posterior_cov = self.posterior_moments(x)
        # add small value to diagonal of cov before sampling
        cov = posterior_cov + jitter * np.eye(posterior_cov.shape[0])
        samples = self._sample(
                mean=posterior_mean.squeeze(),
                cov=cov,
                size=size
        )
        return (samples, np.sqrt(np.diag(cov))) if return_std else samples

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

    def log_marginal_likelihood(self, theta=None, jitter=0):
        """
        Evaluate log likelihood on training data with parameter `theta`

        Args:
            theta (iterable):
            jitter (float): Small number to add to diag of cov to improve numerics

        Returns:

        """
        if not self._is_fit:
            raise ValueError("Need to fit the process before evaluating log likelihood")

        if theta is None:
            kernel = self.kernel
        else:
            kernel = self.kernel.copy_with(*theta)

        cov = kernel(self.train_x, self.train_x) + self.noise_level ** 2 * np.eye(self.train_x.shape[0])

        return log_likelihood(
            mean=0,
            cov=cov,
            t=self.train_y,
            jitter=jitter
        )

    def log_posterior_predictive_likelihood(self, x, y, jitter=0):
        """
        Log posterior predictive likelihood for `y` at test points `x` under this model.

        Args:
            x (np.array): Input Data.
            y (np.array): Output Value.
            jitter (float): Small number to add to diag of cov to improve numerics
        Returns:
            (float): The log posterior predictive likelihood.
        """
        mean, cov = self.posterior_moments(x)
        return log_likelihood(
            mean=mean,
            cov=cov,
            t=y,
            jitter=jitter
        )

    def optimise(self, initial_values, bounds, **kwargs):
        """
        Run optimisation and set hyperparameters of kernel to their MLE values
        by running an optimisation on the *negative* log-likelihood. The model
        is re-fit with the optimal parameter configuration.

        Args:
            initial_values (list): Initial values for the hyperparameters
            bounds (List[tuple]): Optimisation bounds, 2-tuples of numeric.
             If you want to fix a parameter then set the bound to None.
            **kwargs: Keywords for scipy.optimize.minimize

        Returns:
            (scipy.optimize.optimize.OptimizeResult): The output from the optimisation.
        """
        def target(x, gp=self):
            return -1 * gp.log_marginal_likelihood(x)

        x0, bound_ = zip(
            *[(x, b) for x, b in zip(initial_values, bounds) if b is not None]
        )

        opt_output = minimize(
            target,
            x0=x0,
            bounds=bound_,
            **kwargs
        )
        # pass back the fixed parameters
        optimised = iter(opt_output.x)
        params = [
            x if b is None else next(optimised) for x, b in zip(initial_values, bounds)
        ]

        self.kernel = self.kernel.copy_with(*params)
        self.fit(
            x=self.train_x,
            y=self.train_y
        )
        return opt_output


def log_likelihood(mean, cov, t, jitter=0):
    """
    log likelihood of t following gaussian dist with mean mean and covariance cov.

    Args:
        mean (np.array): The mean of the distribution.
        cov (np.array): The covariance matrix for the distribution.
        t (np.array): Test values.
        jitter (float): Small number to add to diag of cov to improve numerics

    Returns:
        log likelihood (float): the log likelihood.
    """
    y = t - mean
    cholesky = np.linalg.cholesky(cov + jitter * np.eye(y.shape[0]))
    first_term = - 0.5 * np.dot(
            y.T,
            cho_solve((cholesky, True), y)
    ).squeeze()
    second_term = - np.sum(np.log(np.diag(cholesky)))
    return first_term + second_term - y.shape[0] * np.log(2 * np.pi) / 2
