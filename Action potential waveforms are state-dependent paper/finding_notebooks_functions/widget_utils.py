"""Shared helpers for the interactive parts of the finding notebooks."""
import contextlib
import io
import warnings

import matplotlib.pyplot as plt
from IPython.display import display

MAX_WIDTH_PX = 1600  # some paper figures are drawn very large; cap their on-screen resolution


def show_figures(plot_fn, *args, quiet=False, **kwargs):
    """Run a plotting function that calls plt.show() itself, and display each new figure exactly once.

    Letting plt.show() fire inside an ipywidgets Output can render figures twice
    (once in the widget, once in the cell), so the call is silenced and the figures shown here.
    quiet=True also hides anything the function prints and any warnings it raises.
    """
    before = set(plt.get_fignums())
    original_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(plt.ioff())
            if quiet:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(warnings.catch_warnings())
                warnings.simplefilter('ignore')
            result = plot_fn(*args, **kwargs)
    finally:
        plt.show = original_show
    for num in [n for n in plt.get_fignums() if n not in before]:
        fig = plt.figure(num)
        fig.set_dpi(min(fig.dpi, MAX_WIDTH_PX / fig.get_figwidth()))
        display(fig)
        plt.close(fig)
    return result
