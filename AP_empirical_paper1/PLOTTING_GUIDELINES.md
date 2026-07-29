# Plotting Guidelines

Conventions used across all spikeparam analysis notebooks. Follow these to keep figures consistent and publication-ready.

---

## Colors

### Spike waveform cluster groups

We use the [Wong 2011](https://www.nature.com/articles/nmeth.1618) colorblind-safe palette for the three waveform cluster groups:

```python
CLUST_COLORS = {
    'Low':  '#0072B2',   # blue
    'Mid':  '#009E73',   # green
    'High': '#D55E00',   # vermillion
}
```

Use these any time you're plotting Low / Mid / High cluster groups against each other — waveform traces, strip plots, box plots, etc.

### LFP features

Each LFP feature has a fixed color used consistently across sliding-window and population-level plots:

```python
feature_shades = {
    'inflection_amp_cluster':  '#c44e52',
    'inflection_time_cluster': '#d97779',
    'peak_amp_cluster':        '#8c564b',
    'peak_sharpness_cluster':  '#a06d62',
    'peak_width_cluster':      '#b38479',
    'exp_lambda_cluster':      '#c561a8',
    'exp_const_cluster':       '#d7aee0',
    'log_isi_cluster':         '#7f7f7f',
    'spk_times_ms_cluster':    '#b0b0b0',
}
```

This dict is defined at the top of each population-level notebook and passed into plotting functions.

---

## Figure style

### Base style
Always call these at the start of a plotting block:
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')          # minimal axes, no gridlines
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})
```

### Despine
Always despine after plotting:
```python
sns.despine(ax=ax)
```

### Figure sizes (inches)
| Use case | Size |
|---|---|
| Single panel | `(5, 4)` |
| Two-panel row | `(10, 4)` |
| 3-panel row | `(14, 4)` |
| Grid of 4 | `(10, 8)` |
| Wide heatmap | `(14, 6)` |
| Correlation matrix | `(8, 7)` |

---

## Labels

### Spike feature display names
Strip `_cluster` suffix in all axis labels, tick labels, and legends. Use a simple helper:

```python
def strip_cluster(name):
    return name.replace('_cluster', '').replace('_', ' ')
```

### LFP feature display names
Use snake_case as-is (e.g. `aperiodic_exp`, `alpha_power`). Do not replace underscores in axis labels — they read cleanly in a monospace font.

### Statistics annotations
Use the standard significance ladder:
```python
def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'
```

---

## Effect size and statistics

- **Effect size**: Cohen's d (pooled SD). Use `_cohens_d()` from `spk_lfp_cluster_comp_analysis.py`.
- **Multiple comparisons**: FDR correction (Benjamini-Hochberg) via `statsmodels.stats.multitest.fdrcorrection`.
- **Non-parametric tests**: Kruskal-Wallis for group comparisons; Spearman for correlations.
- **Resampling**: Bootstrap at the cell level (resample cells with replacement, n=1000). See `bootstrap_population_stats()`.

---

## Heatmaps

- Use `cmap='RdBu_r'` for signed effect sizes (positive = high cluster → higher LFP).
- Use `cmap='Purples'` or `cmap='YlOrRd'` for unsigned magnitudes (yield, |Cohen's d|).
- Always include a colorbar with a label.
- Center diverging colormaps at 0: `vmin=-vmax`.

---

## Grand average trace plots

For peri-spike LFP traces (Low / Mid / High):
- Solid line = grand mean across cells
- Shaded band = ±1 SEM across cells
- Grey `axvspan` = significant time windows
- x-axis: time relative to spike (ms), labeled as `Time from spike (ms)`
- Vertical dashed line at x=0: `ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)`

---

## Saving figures

```python
fig.savefig('figure_name.pdf', bbox_inches='tight', dpi=300)
fig.savefig('figure_name.png', bbox_inches='tight', dpi=150)
```

Do not commit generated figure files (`.png`, `.pdf`) to the repo — these are outputs, not source. See `.gitignore`.
