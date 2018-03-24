#!/usr/bin/env python
"""
Plotting functionality and helper class
"""
#TODO: Docstrings

import matplotlib.pyplot as plt
from scipy.stats import norm


def kill_ticks(ax=None):
    """remove all axes markers"""
    if ax is None:
        ax = plt.gca()
    ax.tick_params(axis='both',
                   which='both',
                   bottom='off',
                   top='off',
                   labelbottom='off',
                   right='off',
                   left='off',
                   labelleft='off')
    return None


def _check_ax(ax, shape=(1,)):
    return plt.subplots(*shape) if ax is None else (ax.figure, ax)


def plot_samples(ax, x, samples, *, stds=None, confidence=0.95, mean=0., label='Sample'):
    """

    Args:
        ax:
        x:
        samples:
        mean:
        stds:
        confidence:
        label (str): label for samples.

    Returns:

    """
    ax.plot(x, samples, label=label)
    if stds is not None:
        if confidence >= 1:
            raise ValueError('confidence must be in [0, 1)')
        point = norm.ppf((1. - confidence) / 2)
        ax.fill_between(x=x.squeeze(),
                        y1=mean - point * stds,
                        y2=mean + point * stds,
                        color='gray',
                        alpha=0.25,
                        label=r'{}\% Confidence'.format(100 * confidence))
    return ax.figure, ax


# TODO: would be good to add error bars, etc.
class GPRPlotter(object):
    """Simple plotting methods for 1-d Gaussian Process Regression"""
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

    def _plot_training_data(self, ax):
        ax.scatter(self.process.train_x, self.process.train_y,
                   marker='+',
                   color='k',
                   linewidths=3,
                   s=300,
                   zorder=100,
                   label='Training Data')
        return None

    def prediction(self, x, *, confidence=0.95, ax=None, legend=True,
                   plot_training_data=True, rc_kwds={}):
        """
        Plot prediction of process at `grid`, conditional on training data.

        Args:
            x:
            confidence:
            ax:
            legend:
            plot_training_data (bool): Optional, default True. Whether to plot training data.
            rc_kwds:

        Returns:

        """
        with plt.rc_context(rc_kwds):
            prediction, stds = self.process.predict(x, return_std=True)
            fig, ax = _check_ax(ax)
            plot_samples(ax=ax,
                         x=x,
                         samples=prediction,
                         stds=stds,
                         confidence=confidence,
                         mean=prediction.squeeze(),
                         label='Prediction')

            if plot_training_data:
                self._plot_training_data(ax=ax)

            ax.grid(alpha=0.1)
            ax.set_title('Prediction', fontsize=25)
            if legend:
                ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
            return fig, ax

    def posterior(self, x, *, size=5, confidence=0.95, ax=None, legend=True,
                  plot_training_data=True, rc_kwds={}):
        """

        Args:
            x:
            size:
            confidence:
            ax:
            legend:
            plot_training_data:
            rc_kwds:

        Returns:

        """
        with plt.rc_context(rc_kwds):
            samples, stds = self.process.sample_posterior(x, size=size, return_std=True)
            prediction = self.process.predict(x, return_std=False)
            fig, ax = _check_ax(ax)
            plot_samples(ax=ax,
                         x=x,
                         samples=samples,
                         stds=stds,
                         confidence=confidence,
                         mean=prediction.squeeze(),
                         label='Sample')

            ax.plot(x, prediction, ls='--', label='Mean', color='gray')

            if plot_training_data:
                self._plot_training_data(ax=ax)

            ax.grid(alpha=0.1)
            ax.set_title('Draws from the Posterior', fontsize=25)
            if legend:
                ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
            return fig, ax

    def prior(self, x, *, size=5, confidence=0.95, ax=None, legend=True, rc_kwds={}):
        """

        Args:
            x:
            size:
            confidence:
            ax:
            legend:
            rc_kwds:

        Returns:

        """
        with plt.rc_context(rc_kwds):
            samples, stds = self.process.sample_prior(x, size=size, return_std=True)
            fig, ax = _check_ax(ax)
            mean = self.process.prior_mean(x)
            plot_samples(ax=ax,
                         x=x,
                         samples=samples,
                         stds=stds,
                         confidence=confidence,
                         mean=mean,
                         label='Sample')

            ax.plot(x, mean, ls='--', label='Prior Mean', color='gray', alpha=0.5)
            ax.grid(alpha=0.1)
            ax.set_title('Draws from the Prior', fontsize=25)
            if legend:
                ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
            return fig, ax
