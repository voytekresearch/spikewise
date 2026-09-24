"""Helpers for 1_parameterization_captures_intra_spike_correlations.ipynb."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

sys.path.append(str(Path(__file__).resolve().parents[1] / 'AP_empirical_paper_all_analyses' / 'datasets' / 'shared_helper_modules'))
from spikewise_plotting import plot_corr_heatmap_only_spk


def _show_once(plot_fn, *args, **kwargs):
    """Run a plotting function that calls plt.show() itself, and display its figure exactly once.

    Letting plt.show() fire inside an ipywidgets Output can render the figure twice
    (once in the widget, once in the cell), so the call is silenced and the figure shown here.
    """
    original_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        with plt.ioff():
            plot_fn(*args, **kwargs)
            fig = plt.gcf()
    finally:
        plt.show = original_show
    display(fig)
    plt.close(fig)


def correlation_matrix_explorer(per_cell, spike_features, dataset='spe-1', cell=14):
    """Dropdown menus to pick a dataset and cell, and plot that cell's feature correlation matrix.

    per_cell : DataFrame with 'dataset', 'cell', 'cell_type' columns plus the spike features.
    """
    cell_options = {
        ds: [(f'Cell {c} ({g.loc[g.cell == c, "cell_type"].iloc[0]})', c) for c in sorted(g.cell.unique())]
        for ds, g in per_cell.groupby('dataset')
    }

    dataset_menu = widgets.Dropdown(options=list(cell_options), value=dataset, description='Dataset:')
    cell_menu = widgets.Dropdown(options=cell_options[dataset], value=cell, description='Cell:')
    plot_area = widgets.Output()

    def show_matrix(*_):
        df_cell = per_cell[(per_cell.dataset == dataset_menu.value) & (per_cell.cell == cell_menu.value)]
        title = (f'{dataset_menu.value}  Cell {cell_menu.value} '
                 f'({df_cell.cell_type.iloc[0]}, n={len(df_cell)} spikes)')
        with plot_area:
            plot_area.clear_output(wait=True)
            _show_once(plot_corr_heatmap_only_spk, df_cell, spike_features, title=title)

    def switch_dataset(change):
        # both datasets have a Cell 1, so the cell value may not change; redraw explicitly, once
        cell_menu.unobserve(show_matrix, names='value')
        cell_menu.options = cell_options[change['new']]
        cell_menu.observe(show_matrix, names='value')
        show_matrix()

    dataset_menu.observe(switch_dataset, names='value')
    cell_menu.observe(show_matrix, names='value')

    display(widgets.HBox([dataset_menu, cell_menu]), plot_area)
    show_matrix()
