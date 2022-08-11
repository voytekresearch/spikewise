"""Test utilities."""


from functools import wraps

import matplotlib.pyplot as plt


def plot_test(func):
    """Decorator for simple testing of plotting functions.
    Notes
    -----
    This decorator closes all plots prior to the test.
    After running the test function, it checks an axis was created with data.
    It therefore performs a minimal test - asserting the plots exists, with no accuracy checking.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        plt.close('all')

        func(*args, **kwargs)

        ax = plt.gca()
        assert ax.has_data()

    return wrapper


def alt_func(sig, fs, weight0=1, weight1=1):
    # Add dummy alt features
    return sig[(len(sig)-1)//2] * weight0 * weight1


def pbar(func, *args, **kwargs):
    # Test dummy progress bar
    return func


def reader(array):
    # Group array reader passthrough
    return array
