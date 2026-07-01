import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import pearsonr
import warnings

# Suppress seaborn palette deprecation noise only
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')
warnings.filterwarnings('ignore', message='.*use_inf_as_na.*')

# Shared color mapping for spike waveform features (used across multiple plot functions)
_FEATURE_COLOR_MAP = {
    'peak_amp':        'C5',
    'exp_const':       'C6',
    'exp_lambda':      'C6',
    'inflection_amp':  'C3',
    'ramp_amp':        'C4',
    'inflection_time': 'C3',
    'peak_sharpness':  'C5',
    'peak_width':      'C5',
    'log_isi':         'C7',
}

def plot_pink_spikes(sp, indices_to_plot):
    """Plot pink noise spike waveforms for the given spike indices."""
    sp.plot(indices_to_plot, color='hotpink', mode='full', show_points=True)
    fig = plt.gcf()
    ax = plt.gca()

    fig.set_size_inches(8, 4)

    for line in ax.get_lines():
        line.set_linewidth(3)

    ax.set_xlabel('Time (ms)', fontsize=26, fontweight='bold')
    ax.set_ylabel('Voltage (mV)', fontsize=26, fontweight='bold')
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([-60, 20])
    ax.tick_params(axis='both', which='major', labelsize=22, width=2.5, color='black')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(2.5)
        ax.spines[spine].set_color('black')

    legend = ax.get_legend()
    if legend:
        _handles = legend.legend_handles
        _labels  = [t.get_text() for t in legend.get_texts()]
        legend.remove()
        legend = ax.legend(_handles, _labels,
                           loc='center left', bbox_to_anchor=(1.01, 0.5),
                           frameon=False)
        plt.setp(legend.get_texts(), fontsize=18, fontweight='bold')
        for handle in legend.legend_handles:
            handle.set_markersize(14)

    plt.tight_layout()
    plt.show()

def set_plot_style():
    """Global bold/cartoony style: big fonts, only left+bottom spines, minimal ticks."""
    plt.rcParams.update({
        'font.family':            'Helvetica Neue',
        'font.size':              18,
        'font.weight':            'bold',
        'axes.labelsize':         20,
        'axes.labelweight':       'bold',
        'axes.titlesize':         20,
        'axes.titleweight':       'bold',
        'xtick.labelsize':        16,
        'ytick.labelsize':        16,
        'legend.fontsize':        16,
        'legend.title_fontsize':  16,
        'figure.titlesize':       22,
        'figure.titleweight':     'bold',
        # clean white background (overrides seaborn whitegrid)
        'axes.facecolor':         'white',
        'figure.facecolor':       'white',
        'axes.edgecolor':         '#1a1a1a',
        # only left + bottom spines
        'axes.spines.top':        False,
        'axes.spines.right':      False,
        'axes.linewidth':         2.5,
        # ticks — thick, minimal
        'xtick.major.width':      2.0,
        'ytick.major.width':      2.0,
        'xtick.major.size':       6,
        'ytick.major.size':       6,
        'xtick.minor.visible':    False,
        'ytick.minor.visible':    False,
        # no grid
        'axes.grid':              False,
    })


_SPIKE_FEATURES = [
    'ramp_amp', 'inflection_time', 'inflection_amp',
    'peak_amp', 'peak_width', 'peak_sharpness',
    'exp_lambda', 'exp_const',
]
_STIM_FEATURES = ['stim_mean', 'stim_std', 'stim_exp']

# Bold palette — one colour per stim feature, reused across subplots
_SCATTER_COLORS = {'stim_mean': '#E07B54', 'stim_std': '#5B8DB8', 'stim_exp': '#72B26C'}


def plot_top_correlations_by_window(df_w, window_ms, top=False, top_n=3):
    """Grid of spike-feature × stim-feature correlations for a given window.

    top=False (default): all combinations, compact style.
                         Rows = stim features, cols = spike features sorted by |r|.
    top=True:            top_n (default 3) spike features per stim feature, cartoon style
                         suitable for figures. 3 rows × top_n cols.
    """
    from scipy.stats import pearsonr as _pearsonr

    # Collect all valid pairs grouped by stim feature
    stf_groups = {}
    for stf in _STIM_FEATURES:
        group = []
        for sf in _SPIKE_FEATURES:
            if sf not in df_w.columns or stf not in df_w.columns:
                continue
            valid = df_w[[sf, stf]].dropna()
            if len(valid) < 10:
                continue
            r, p = _pearsonr(valid[sf], valid[stf])
            group.append((abs(r), r, p, sf, valid))
        group.sort(key=lambda x: x[0], reverse=True)
        if group:
            stf_groups[stf] = group

    if not stf_groups:
        print("No valid pairs found.")
        return

    def _draw_ax_compact(ax, abs_r, r, p, sf, stf, valid):
        color = _FEATURE_COLOR_MAP.get(sf, '#888')
        ax.scatter(valid[sf], valid[stf], s=60, alpha=0.55, color=color, linewidths=0)
        m, b = np.polyfit(valid[sf], valid[stf], 1)
        xs = np.linspace(valid[sf].min(), valid[sf].max(), 200)
        ax.plot(xs, m * xs + b, color='#1a1a1a', lw=3, zorder=3)
        star  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ann_x, ann_y = (0.04, 0.96) if r >= 0 else (0.04, 0.04)
        ann_va = 'top' if r >= 0 else 'bottom'
        ax.text(ann_x, ann_y, f'r = {r:.2f}',
                transform=ax.transAxes, ha='left', va=ann_va,
                fontsize=22, fontweight='bold', color='#1a1a1a',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.85),
                zorder=5)
        star_y = (ann_y - 0.22) if r >= 0 else (ann_y + 0.22)
        ax.text(ann_x, star_y, star,
                transform=ax.transAxes, ha='left', va=ann_va,
                fontsize=40, fontweight='bold', color='#1a1a1a', zorder=5)
        ax.set_xlabel(sf.replace('_', ' '), fontsize=22, fontweight='bold')
        ax.set_ylabel(stf.replace('_', ' ').replace('std', 'stdev'), fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', labelsize=20, width=2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        ax.locator_params(nbins=3)

    def _draw_ax_cartoon(ax, abs_r, r, p, sf, stf, valid, first_col=False):
        color = _FEATURE_COLOR_MAP.get(sf, '#888')
        ax.scatter(valid[sf], valid[stf], s=200, alpha=0.75, color=color,
                   linewidths=0, zorder=2)
        m, b = np.polyfit(valid[sf], valid[stf], 1)
        xs = np.linspace(valid[sf].min(), valid[sf].max(), 200)
        ax.plot(xs, m * xs + b, color='#1a1a1a', lw=5, zorder=3)
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        if sf == 'inflection_amp' and stf == 'stim_std':
            ann_x, ann_y, ann_ha, ann_va = 0.96, 0.96, 'right', 'top'
        elif r >= 0:
            ann_x, ann_y, ann_ha, ann_va = 0.04, 0.96, 'left', 'top'
        else:
            ann_x, ann_y, ann_ha, ann_va = 0.04, 0.04, 'left', 'bottom'
        ax.text(ann_x, ann_y, f'r = {r:.2f}',
                transform=ax.transAxes, ha=ann_ha, va=ann_va,
                fontsize=80, fontweight='bold', color='#1a1a1a',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.9),
                zorder=5)
        star_y = (ann_y - 0.22) if ann_va == 'top' else (ann_y + 0.22)
        ax.text(ann_x, star_y, star,
                transform=ax.transAxes, ha=ann_ha, va=ann_va,
                fontsize=140, fontweight='bold', color='#1a1a1a', zorder=5)
        ax.set_xlabel(sf.replace('_', ' '), fontsize=84, fontweight='bold')
        if first_col:
            ax.set_ylabel(stf.replace('_', ' ').replace('std', 'stdev'), fontsize=84, fontweight='bold')
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='both', labelsize=80, width=4, length=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(5)
        ax.spines['bottom'].set_linewidth(5)
        ax.locator_params(nbins=3)

    if top:
        ncols = top_n
        nrows = len([stf for stf in _STIM_FEATURES if stf in stf_groups])
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(12 * ncols, 12 * nrows),
                                 constrained_layout=True)
        axes = np.array(axes).reshape(nrows, ncols)
        row_idx = 0
        for stf in _STIM_FEATURES:
            if stf not in stf_groups:
                continue
            for col_idx, (abs_r, r, p, sf, valid) in enumerate(stf_groups[stf][:top_n]):
                _draw_ax_cartoon(axes[row_idx, col_idx], abs_r, r, p, sf, stf, valid,
                                 first_col=(col_idx == 0))
            for col_idx in range(len(stf_groups[stf][:top_n]), ncols):
                axes[row_idx, col_idx].set_visible(False)
            row_idx += 1
        fig.suptitle(f'Top {top_n} spike × stim correlations — {window_ms} ms window',
                     fontsize=28, fontweight='bold')
    else:
        nrows = len([stf for stf in _STIM_FEATURES if stf in stf_groups])
        ncols = max(len(g) for g in stf_groups.values())
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(7 * ncols, 7 * nrows),
                                 constrained_layout=True)
        axes = np.array(axes).reshape(nrows, ncols)
        row_idx = 0
        for stf in _STIM_FEATURES:
            if stf not in stf_groups:
                continue
            group = stf_groups[stf]
            for col_idx, (abs_r, r, p, sf, valid) in enumerate(group):
                _draw_ax_compact(axes[row_idx, col_idx], abs_r, r, p, sf, stf, valid)
            for col_idx in range(len(group), ncols):
                axes[row_idx, col_idx].set_visible(False)
            row_idx += 1
        fig.suptitle(f'All spike × stim correlations — {window_ms} ms window',
                     fontsize=24, fontweight='bold')

    plt.show()

def plot_avg_waveform_by_stim_type(all_constant_spks, all_ramp_spks, all_pink_spks):
    """Mean ± SD spike waveform for each stimulus type. Pass None to skip a type."""
    plt.figure(figsize=(10, 6))

    if all_constant_spks is not None:
        mean_c = np.mean(all_constant_spks, axis=0)
        std_c  = np.std(all_constant_spks, axis=0)
        plt.plot(mean_c, label='Mean Constant Spikes', color='green', linewidth=4)
        plt.fill_between(range(len(mean_c)), mean_c - std_c, mean_c + std_c, color='green', alpha=0.3)

    if all_ramp_spks is not None:
        mean_r = np.mean(all_ramp_spks, axis=0)
        std_r  = np.std(all_ramp_spks, axis=0)
        plt.plot(mean_r, label='Mean Ramp Spikes', color='purple', linewidth=4)
        plt.fill_between(range(len(mean_r)), mean_r - std_r, mean_r + std_r, color='purple', alpha=0.3)

    if all_pink_spks is not None:
        mean_p = np.mean(all_pink_spks, axis=0)
        std_p  = np.std(all_pink_spks, axis=0)
        plt.plot(mean_p, label='Mean Pink Spikes', color='hotpink', linewidth=4)
        plt.fill_between(range(len(mean_p)), mean_p - std_p, mean_p + std_p, color='lightpink', alpha=0.3)

    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage')
    plt.xlim(1600, 2400)
    plt.show()



def plot_confusion_matrix(best_model, X_test, y_test):
    """Confusion matrix for a trained classifier on test data."""
    from sklearn.metrics import confusion_matrix
    cm      = confusion_matrix(y_test, best_model.predict(X_test), labels=best_model.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)

    n_classes = len(best_model.classes_)
    fig_size  = 5 + 2 * n_classes
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    display.plot(cmap='Blues', ax=ax, xticks_rotation=0, values_format='d')
    ax.set_aspect('equal')

    for text in ax.texts:
        text.set_fontsize(28)
        text.set_fontweight('bold')

    ax.set_xlabel('Predicted Label', fontsize=24, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=22, width=2.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    # confusion matrix needs all four spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color('#1a1a1a')
    plt.tight_layout()
    plt.show()

def plot_ridge_results(y, ridge_results, title="Ridge Regression Results"):
    """Actual vs predicted scatter, coefficient bar chart, bootstrapped distributions for ridge results."""
    y_pred_cv = ridge_results["y_pred_cv"]
    coefficients = ridge_results["coefficients"]
    feature_names = ridge_results["feature_names"]
    ci_lower = ridge_results["ci_lower"]
    ci_upper = ridge_results["ci_upper"]
    p_values = ridge_results["p_values"]
    r2_scores = ridge_results["r2_scores"]
    adjusted_r2_scores = ridge_results["adjusted_r2_scores"]
    bootstrapped_r2 = ridge_results["bootstrapped_r2"]
    bootstrapped_adjusted_r2 = ridge_results["bootstrapped_adjusted_r2"]
    r2_mean = ridge_results["r2_mean"]
    r2_ci = ridge_results["r2_ci"]
    adjusted_r2_mean = ridge_results["adjusted_r2_mean"]
    adjusted_r2_ci = ridge_results["adjusted_r2_ci"]

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names, 
        'Coefficient': coefficients,
        'CI Lower': ci_lower,
        'CI Upper': ci_upper,
        'p-value': p_values
    })
    
    feature_importance_df = feature_importance_df.sort_values(by='Coefficient', ascending=False)

    feature_importance_df['Color'] = feature_importance_df['Feature'].apply(
        lambda x: 'hotpink' if x.startswith('stim_') else _FEATURE_COLOR_MAP.get(x, 'gray')
    )

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred_cv, s=55)
    plt.xlabel("Actual Values", fontsize=20)
    plt.ylabel("Predicted Values", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    m, b = np.polyfit(y, y_pred_cv, 1)
    plt.plot(y, m * y + b, color='red', linewidth=2)  
    plt.show()

    # ── Beta weight bar chart with 95% CI error bars ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = feature_importance_df['Color'].tolist()
    xerr_low  = feature_importance_df['Coefficient'].values - feature_importance_df['CI Lower'].values
    xerr_high = feature_importance_df['CI Upper'].values   - feature_importance_df['Coefficient'].values
    ax.barh(feature_importance_df['Feature'], feature_importance_df['Coefficient'],
            xerr=[xerr_low, xerr_high], color=colors, alpha=0.8,
            error_kw=dict(ecolor='#333', lw=1.5, capsize=4))
    ax.axvline(0, color='gray', lw=1, ls='--', alpha=0.7)

    # Add significance stars
    for idx, row in feature_importance_df.reset_index(drop=True).iterrows():
        p = row['p-value']
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        if star:
            x_pos = row['CI Upper'] + abs(feature_importance_df['Coefficient'].max()) * 0.03
            ax.text(x_pos, idx, star, va='center', fontsize=12, color='#D55E00', fontweight='bold')

    ax.set_xlabel('Beta weight (standardised)', fontsize=13)
    ax.set_ylabel('Feature', fontsize=13)
    plt.tight_layout()
    plt.show()

    # ── Beta weight summary table ─────────────────────────────────────────────
    def _fmt_p(p):
        if p < 0.0001: return '< 0.0001'
        return f'{p:.4f}'
    def _stars(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'ns'

    table = feature_importance_df[["Feature", "Coefficient", "CI Lower", "CI Upper", "p-value"]].copy()
    table['sig'] = table['p-value'].apply(_stars)
    table['p-value'] = table['p-value'].apply(_fmt_p)
    table.columns = ['Feature', 'Beta', 'CI 2.5%', 'CI 97.5%', 'p (bootstrap)', 'sig']
    print('\nBeta weights (standardised, 95% bootstrap CI):')
    print(table.to_string(index=False))

    bootstrapped_coefs = ridge_results["bootstrapped_coefs"]
    
    plt.figure(figsize=(12, 6))
    
    coef_df = pd.DataFrame(bootstrapped_coefs, columns=feature_names)
    coef_df = coef_df.melt(var_name='Feature', value_name='Coefficient Value')
    
    coef_df['Color'] = coef_df['Feature'].apply(
        lambda x: 'hotpink' if x.startswith('stim_') else _FEATURE_COLOR_MAP.get(x, 'gray')
    )
    
    color_dict = coef_df.groupby('Feature')['Color'].first().to_dict()
    sns.violinplot(
        x='Feature',
        y='Coefficient Value',
        data=coef_df,
        hue='Feature',
        palette=color_dict,
        inner="point",
        density_norm="width",
        legend=False
    )
    
    plt.xticks(rotation=45)
    plt.ylabel("Coefficient Value", fontsize=16)
    plt.xlabel("Feature", fontsize=16)
    plt.show()

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.boxplot(x=r2_scores)
    plt.xlabel("R²", fontsize=14)

    plt.subplot(1, 3, 2)
    sns.histplot(bootstrapped_r2, kde=True, bins=30)
    plt.axvline(r2_mean, color='red', linestyle='--', label=f"Mean R²: {r2_mean:.3f}")
    plt.axvline(r2_ci[0], color='gray', linestyle=':', label=f"95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")
    plt.axvline(r2_ci[1], color='gray', linestyle=':')
    plt.xlabel("R²", fontsize=14)
    plt.legend()

    plt.subplot(1, 3, 3)
    sns.histplot(bootstrapped_adjusted_r2, kde=True, bins=30)
    plt.axvline(adjusted_r2_mean, color='red', linestyle='--', label=f"Mean Adjusted R²: {adjusted_r2_mean:.3f}")
    plt.axvline(adjusted_r2_ci[0], color='gray', linestyle=':', label=f"95% CI: [{adjusted_r2_ci[0]:.3f}, {adjusted_r2_ci[1]:.3f}]")
    plt.axvline(adjusted_r2_ci[1], color='gray', linestyle=':')
    plt.xlabel("Adjusted R²", fontsize=14)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"\nMean Adjusted R-squared (Bootstrapped): {adjusted_r2_mean:.3f}")
    print(f"95% CI for Adjusted R-squared: {adjusted_r2_ci}")


def plot_ridge_results_grid(results_dict, keys, labels=None, ys=None):
    """Show scatter, beta-weight, and R² panels for multiple ridge fits side-by-side.

    results_dict : {key: ridge_results}
    keys         : ordered list of keys to plot (columns)
    labels       : display title per column (defaults to keys)
    ys           : {key: y_series} – required for the scatter row
    """
    n = len(keys)
    if labels is None:
        labels = list(keys)

    # ── Scatter: actual vs predicted ─────────────────────────────────────────
    if ys is not None or any('y_true' in results_dict[k] for k in keys):
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), constrained_layout=True)
        if n == 1:
            axes = [axes]
        for ax, key, label in zip(axes, keys, labels):
            res  = results_dict[key]
            yhat = np.asarray(res['y_pred_cv'])
            if 'y_true' in res:
                y = res['y_true']
            elif ys is not None:
                y = np.asarray(ys[key])
                if len(y) != len(yhat):
                    print(f"Warning: y size {len(y)} != y_pred_cv size {len(yhat)} for '{key}' — skipping scatter.")
                    continue
            else:
                continue
            r2   = res['r2_mean']
            ci   = res['r2_ci']
            ax.scatter(y, yhat, s=35, alpha=0.55)
            m, b = np.polyfit(y, yhat, 1)
            ax.plot(y, m * y + b, color='red', lw=2.5)
            ax.set_xlabel(f'Actual  ({label})', fontsize=22, fontweight='bold')
            ax.set_ylabel('Predicted', fontsize=22, fontweight='bold')
            ax.text(0.05, 0.95, f'R²={r2:.3f}\n[{ci[0]:.3f},{ci[1]:.3f}]',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=18, fontweight='bold')
            ax.tick_params(axis='both', labelsize=18, width=2)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontweight('bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.locator_params(nbins=3)
        plt.show()

    # ── Beta weights: shared feature order, only left col gets y-labels ──────
    # Build a stable feature order (sorted by mean |coef| across all keys)
    feat_importance = {}
    for key in keys:
        res = results_dict[key]
        for feat, coef in zip(res['feature_names'], res['coefficients']):
            feat_importance[feat] = feat_importance.get(feat, 0) + abs(coef)
    ordered_feats = sorted(feat_importance, key=lambda x: feat_importance[x])
    n_feats = len(ordered_feats)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, max(5, n_feats * 0.38 + 1)),
                             sharey=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    for col, (ax, key, label) in enumerate(zip(axes, keys, labels)):
        res = results_dict[key]
        feat_df = pd.DataFrame({
            'Feature':     res['feature_names'],
            'Coefficient': res['coefficients'],
            'CI Lower':    res['ci_lower'],
            'CI Upper':    res['ci_upper'],
            'p-value':     res['p_values'],
        }).set_index('Feature').reindex(ordered_feats).dropna()

        colors    = ['hotpink' if f.startswith('stim_') else _FEATURE_COLOR_MAP.get(f, 'gray')
                     for f in feat_df.index]
        xerr_low  = feat_df['Coefficient'] - feat_df['CI Lower']
        xerr_high = feat_df['CI Upper']    - feat_df['Coefficient']

        ax.barh(feat_df.index, feat_df['Coefficient'],
                xerr=[xerr_low.values, xerr_high.values],
                color=colors, alpha=0.82,
                edgecolor='#1a1a1a', linewidth=1.8,
                error_kw=dict(ecolor='#333', lw=2, capsize=5))
        ax.axvline(0, color='gray', lw=1, ls='--', alpha=0.7)

        x_max = float(feat_df['CI Upper'].max())
        for i, (feat, row) in enumerate(feat_df.iterrows()):
            p    = row['p-value']
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            if star:
                ax.text(x_max + abs(x_max) * 0.06, i, star,
                        va='center', fontsize=20, color='#D55E00', fontweight='bold')

        ax.set_xlabel(f'Beta weight (std.)  —  {label}', fontsize=22, fontweight='bold')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=18, width=2)
        ax.tick_params(axis='y', labelsize=18, width=2)
        ax.locator_params(axis='x', nbins=3)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
    plt.show()

    # ── Bootstrapped R² distributions ────────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for col, (ax, key, label) in enumerate(zip(axes, keys, labels)):
        res    = results_dict[key]
        boot   = res['bootstrapped_r2']
        r2m    = res['r2_mean']
        ci     = res['r2_ci']
        p_perm = res.get('p_val_perm', None)
        ax.hist(boot, bins=30, alpha=0.7, color='steelblue', edgecolor='none')
        ax.axvline(r2m,   color='red',  ls='--', lw=2,   label=f'Mean={r2m:.3f}')
        ax.axvline(ci[0], color='gray', ls=':',  lw=1.5)
        ax.axvline(ci[1], color='gray', ls=':',  lw=1.5,
                   label=f'95% CI [{ci[0]:.3f},{ci[1]:.3f}]')
        title = label
        if p_perm is not None:
            sig   = ' *' if p_perm < 0.05 else ''
            title += f'\np={p_perm:.3f}{sig}'
        ax.set_xlabel('R²', fontsize=22, fontweight='bold')
        ax.set_ylabel('Count' if col == 0 else '', fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', labelsize=18, width=2)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.locator_params(nbins=3)
        leg = ax.legend(frameon=False, loc='upper left', fontsize=16)
        for t in leg.get_texts():
            t.set_fontweight('bold')
    plt.show()


# --- Define the function to plot R-squared comparison ---
def plot_r2_comparison(ridge_results_spike_only, ridge_results_spike_stim, ridge_results_stim_only):
    feature_sets = ["No stim features", "With stim features", "Only stim features"]
    # Names of the models
    labels = ['Average r-squared', 'Adjusted r-squared']
    no_stim = [ridge_results_spike_only["r2_mean"], ridge_results_spike_only["adjusted_r2_mean"]]
    with_stim = [ridge_results_spike_stim["r2_mean"],ridge_results_spike_stim["adjusted_r2_mean"] ]
    only_stim = [ridge_results_stim_only["r2_mean"], ridge_results_stim_only["adjusted_r2_mean"]]
  
    width = 0.3
    
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, no_stim, width, label='No stim features', color='#5b83bc')
    rects2 = ax.bar(x, with_stim, width, label='With stim features (Combined)', color='#974a75')
    rects3 = ax.bar(x + width, only_stim, width, label='Only stim features', color='#df8ac1')
    
    # Adding titles and labels
    
    ax.set_ylabel('R-squared', fontsize=20, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold')
    
    # Set the y-ticks to specific values to reduce clutter
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4'], fontsize=14)
    
    # Adding text labels on the bars
    def autolabel(rects):
        """Attach a text label above each bar displaying its height, formatted to three decimal places."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    ax.legend(fontsize=16, frameon=True, facecolor='white', framealpha=1)
    
    # Removing the top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Show the plot
    #plt.ylim(0, 0.2)
    plt.show()


# --- Define the function to plot Actual vs Predicted log ISI ---
def plot_combined_actual_vs_predicted(y_actual, all_features_y_pred, all_features_and_stim_y_pred, only_stim_y_pred):
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # First set: all_features (No stim features)
    sns.scatterplot(x=y_actual, y=all_features_y_pred, s=55, label='No stim features', color='#5b83bc')
    m, b = np.polyfit(y_actual, all_features_y_pred, 1)
    plt.plot(y_actual, m * y_actual + b, color='#5b83bc', linewidth=2)
    
    # Second set: all_features_and_stim (With stim features)
    sns.scatterplot(x=y_actual, y=all_features_and_stim_y_pred, s=55, label='With stim features', color='#974a75')
    m, b = np.polyfit(y_actual, all_features_and_stim_y_pred, 1)
    plt.plot(y_actual, m * y_actual + b, color='#974a75', linewidth=2)
    
    # Third set: only_stim (Only stim features)
    sns.scatterplot(x=y_actual, y=only_stim_y_pred, s=55, label='Only stim features', color='#df8ac1')
    m, b = np.polyfit(y_actual, only_stim_y_pred, 1)
    plt.plot(y_actual, m * y_actual + b, color='#df8ac1', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Actual log_isi', fontsize=20)
    plt.ylabel('Predicted log_isi', fontsize=20)
    plt.xticks(fontsize=16)               
    plt.yticks(fontsize=16)
    
    # Add legend
    plt.legend([], [], frameon=False)
    
    # Show the plot
    plt.show()


# --- Define the function to plot Feature Importance ---
def plot_combined_feature_importance(all_features_feature_importance_df, all_features_and_stim_feature_importance_df,only_stim_feature_importance_df):
    # Combine the three DataFrames
    combined_feature_importance_df = pd.concat([
        all_features_feature_importance_df,
        all_features_and_stim_feature_importance_df,
        only_stim_feature_importance_df
    ])
    
    # Plot combined feature importance without the legend
    plt.figure(figsize=(12, 8))  # Increase the figure size
    sns.barplot(x='Coefficient', y='Feature', hue='Model', data=combined_feature_importance_df, 
                palette=['#5b83bc', '#974a75', '#df8ac1'], linewidth=0.5)  # Increased bar thickness
    
    # Add labels and title
    plt.xlabel('Coefficient (Importance)', fontsize=20)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=20)  # Increase font size for the feature names
    
    # Disable the legend
    plt.legend([], [], frameon=False)
    
    # Show the plot
    plt.show()


# Function to plot feature importance
def plot_feature_importance_categorical(best_model, X, bootstrap_importances):
    """Plot RF feature importances as mean ± 95% CI across bootstrap runs."""
    feature_names = X.columns
    imp_mean  = np.mean(bootstrap_importances, axis=0)
    imp_lower = np.percentile(bootstrap_importances, 2.5,  axis=0)
    imp_upper = np.percentile(bootstrap_importances, 97.5, axis=0)

    df_importance = pd.DataFrame({
        'Feature':  feature_names,
        'Importance': imp_mean,
        'CI Lower': imp_lower,
        'CI Upper': imp_upper,
    })
    df_importance = df_importance.sort_values(by='Importance', ascending=True)  # ascending=True puts largest at top for barh
    df_importance['Color'] = df_importance['Feature'].map(_FEATURE_COLOR_MAP).fillna('#8c8c8c')
    df_importance['Label'] = df_importance['Feature'].str.replace('_', ' ')

    xerr_low  = df_importance['Importance'].values - df_importance['CI Lower'].values
    xerr_high = df_importance['CI Upper'].values   - df_importance['Importance'].values

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(df_importance['Label'], df_importance['Importance'],
                   xerr=[xerr_low, xerr_high],
                   color=df_importance['Color'].tolist(),
                   edgecolor='#1a1a1a', linewidth=2.5,
                   error_kw={'elinewidth': 4, 'capsize': 8, 'capthick': 4, 'ecolor': '#1a1a1a'})
    ax.set_xlabel('Importance', fontsize=36, fontweight='bold')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=32, width=2.5)
    ax.tick_params(axis='y', labelsize=38, width=2.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    plt.tight_layout()
    plt.show()

# Function to plot bootstrapped accuracies as histograms
def plot_bootstrap_histograms(bootstrapped_results, model_names, overlay=False):
    _FS_AX  = 28
    _FS_TK  = 24
    _LW_SP  = 2.5
    _COLORS = ['#AED6F1', '#2980B9', '#1A5276']  # light → dark blue

    all_accs = np.concatenate(bootstrapped_results)
    x_pad = (all_accs.max() - all_accs.min()) * 0.05
    xlim = (all_accs.min() - x_pad, all_accs.max() + x_pad)

    def _style_ax(ax):
        ax.set_xlim(xlim)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax.tick_params(axis='both', labelsize=_FS_TK, width=_LW_SP)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_linewidth(_LW_SP)
            spine.set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if overlay:
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        # Minimum absolute bandwidth: jitter near-zero-variance distributions
        _global_range = all_accs.max() - all_accs.min()
        _min_bw = max(_global_range * 0.02, 0.005)
        for accs, model_name, col in zip(bootstrapped_results, model_names, _COLORS):
            accs_mean = np.mean(accs)
            _std = np.std(accs)
            _accs_plot = (accs + np.random.default_rng(42).normal(0, _min_bw, size=len(accs))
                          if _std < _min_bw else accs)
            sns.kdeplot(_accs_plot, ax=ax, color=col, lw=3.5, fill=True, alpha=0.45,
                        label=f'{model_name}  (mean={accs_mean:.3f})')
            ax.axvline(accs_mean, color=col, ls='--', lw=2.5, zorder=5)
        ax.set_xlabel('Bootstrap accuracy', fontsize=_FS_AX, fontweight='bold', labelpad=10)
        ax.set_ylabel('Density', fontsize=_FS_AX, fontweight='bold', labelpad=10)
        ax.set_title('Bootstrap accuracy distributions', fontsize=_FS_AX, fontweight='bold', pad=14)
        leg = ax.legend(frameon=False, fontsize=_FS_TK,
                        prop={'size': _FS_TK, 'weight': 'bold'},
                        loc='upper center', bbox_to_anchor=(0.5, -0.18),
                        ncol=1)
        _style_ax(ax)
    else:
        n = len(bootstrapped_results)
        fig, axes = plt.subplots(1, n, figsize=(8 * n, 7), constrained_layout=True)
        if n == 1:
            axes = [axes]
        _global_range = all_accs.max() - all_accs.min()
        _min_bw = max(_global_range * 0.02, 0.005)
        for ax, accs, model_name, col in zip(axes, bootstrapped_results, model_names, _COLORS):
            accs_mean = np.mean(accs)
            accs_ci   = np.percentile(accs, [2.5, 97.5])
            _std = np.std(accs)
            _accs_plot = (accs + np.random.default_rng(42).normal(0, _min_bw, size=len(accs))
                          if _std < _min_bw else accs)
            sns.histplot(_accs_plot, kde=True, bins=30, ax=ax,
                         color='#0072B2', alpha=0.65, edgecolor='white',
                         line_kws={'lw': 3.0, 'color': '#0072B2'})
            ax.axvline(accs_mean, color='black', ls='--', lw=3.0, zorder=5)
            ax.axvline(accs_ci[0], color='black', ls=':', lw=2.0, zorder=5)
            ax.axvline(accs_ci[1], color='black', ls=':', lw=2.0, zorder=5)
            xlim_mid = (xlim[0] + xlim[1]) / 2
            txt_x, txt_ha = (0.97, 'right') if accs_mean < xlim_mid else (0.03, 'left')
            ax.text(txt_x, 0.97, f'Mean = {accs_mean:.3f}',
                    transform=ax.transAxes, ha=txt_ha, va='top',
                    fontsize=_FS_AX + 4, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.0, pad=4))
            ax.set_xlabel('Bootstrap accuracy', fontsize=_FS_AX, fontweight='bold', labelpad=10)
            ax.set_ylabel('Count', fontsize=_FS_AX, fontweight='bold', labelpad=10)
            ax.set_title(model_name, fontsize=_FS_AX, fontweight='bold', pad=14)
            _style_ax(ax)

    plt.show()


def plot_window_expansion(results_by_window, windows_ms,
                          targets=('stim_mean', 'stim_std', 'stim_exp'),
                          target_labels=('stim mean', 'stim std', 'stim exp'),
                          title='',
                          shuffle_results=None,
                          results_by_window_2=None,
                          shuffle_results_2=None,
                          cell_labels=('Cell 1', 'Cell 2')):
    """
    Line plot of bootstrap R² (mean ± 95% CI shaded band) across window sizes.
    Filled circles = permutation p<0.05; open circles = n.s.
    Shuffle shown as dashed line with shaded band.
    """
    _FS_AX   = 28
    _FS_TK   = 24
    _FS_STR  = 36
    _LW_SP   = 2.5
    _LW_LINE = 3.5
    _COL_SIG = '#D55E00'
    _COL_SHF = '#56B4E9'

    two_cells = results_by_window_2 is not None
    n_targets = len(targets)

    fig, axes = plt.subplots(1, n_targets, figsize=(8 * n_targets, 7))
    if n_targets == 1:
        axes = [axes]
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.08, right=0.97, wspace=0.35)

    def _extract(results, tgt):
        r2s, lo, hi, sigs = [], [], [], []
        for wms in windows_ms:
            key = f'{wms}ms_{tgt}'
            if key not in results:
                r2s.append(np.nan); lo.append(np.nan); hi.append(np.nan); sigs.append(False)
                continue
            res = results[key]
            r2s.append(res['r2_mean'])
            lo.append(res['r2_ci'][0])
            hi.append(res['r2_ci'][1])
            sigs.append(res.get('p_val_perm', 1.0) < 0.05)
        return (np.array(r2s, dtype=float), np.array(lo, dtype=float),
                np.array(hi, dtype=float), sigs)

    def _draw_line(ax, xs, r2s, lo, hi, sigs, color, label='', lw=_LW_LINE,
                   ls='-'):
        ax.plot(xs, r2s, color=color, lw=lw, ls=ls, zorder=3,
                label=label, solid_capstyle='round')
        ax.errorbar(xs, r2s,
                    yerr=[r2s - lo, hi - r2s],
                    fmt='none', color=color, capsize=7, capthick=2.5,
                    lw=2.0, zorder=4)
        for xi, (r2v, sig) in enumerate(zip(r2s, sigs)):
            if np.isnan(r2v):
                continue
            if sig:
                ax.scatter(xs[xi], r2v, s=110, color=color,
                           edgecolors='black', linewidths=1.5, zorder=5)
            else:
                ax.scatter(xs[xi], r2v, s=110, facecolors='white',
                           edgecolors=color, linewidths=2.0, zorder=5)
        for xi, (hv, sig) in enumerate(zip(hi, sigs)):
            if sig and not np.isnan(hv):
                ax.text(xs[xi], hv + 0.012, '*',
                        ha='center', va='bottom', fontsize=_FS_STR,
                        fontweight='bold', color=color)

    xs = np.arange(len(windows_ms))

    for ax, tgt, tgt_lbl in zip(axes, targets, target_labels):
        r2s1, lo1, hi1, sigs1 = _extract(results_by_window, tgt)
        lbl1 = cell_labels[0] if two_cells else ''
        _draw_line(ax, xs, r2s1, lo1, hi1, sigs1, color=_COL_SIG, label=lbl1)

        if two_cells:
            r2s2, lo2, hi2, sigs2 = _extract(results_by_window_2, tgt)
            _draw_line(ax, xs, r2s2, lo2, hi2, sigs2,
                       color='#0072B2', label=cell_labels[1])

        if shuffle_results is not None:
            sr2s, slo, shi, ssigs = _extract(shuffle_results, tgt)
            shuf_lbl = f'shuffle ({cell_labels[0]})' if two_cells else 'shuffle'
            _draw_line(ax, xs, sr2s, slo, shi, ssigs,
                       color=_COL_SHF, label=shuf_lbl, lw=2.5, ls='--')

        if shuffle_results_2 is not None and two_cells:
            sr2s2, slo2, shi2, ssigs2 = _extract(shuffle_results_2, tgt)
            _draw_line(ax, xs, sr2s2, slo2, shi2, ssigs2,
                       color='#AAAAAA', label=f'shuffle ({cell_labels[1]})', lw=2.5, ls='--')

        ax.axhline(0, color='black', lw=1.8, ls='--', alpha=0.4, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(w) for w in windows_ms],
                           rotation=45, ha='right', fontsize=_FS_TK, fontweight='bold')
        ax.tick_params(axis='y', labelsize=_FS_TK, width=_LW_SP)
        ax.tick_params(axis='x', width=_LW_SP)
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight('bold')
        ax.set_xlabel('Pre-spike window (ms)', fontsize=_FS_AX, fontweight='bold', labelpad=10)
        ax.set_ylabel('Bootstrap R²', fontsize=_FS_AX, fontweight='bold', labelpad=10)
        ax.set_title(tgt_lbl, fontsize=_FS_AX, fontweight='bold', pad=14)
        ax.locator_params(axis='y', nbins=4)
        for spine in ax.spines.values():
            spine.set_linewidth(_LW_SP)
            spine.set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # legend from first axis handles
    handles, labels_ = axes[0].get_legend_handles_labels()
    # add sig/ns dot legend entries
    handles += [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555',
               markeredgecolor='black', markersize=10, label='p < 0.05 (perm.)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#555555', markersize=10, markeredgewidth=2,
               label='n.s.'),
    ]
    labels_ += ['p < 0.05 (perm.)', 'n.s.']
    fig.legend(handles, labels_, frameon=False,
               loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), fontsize=_FS_TK - 2,
               prop={'size': _FS_TK - 2, 'weight': 'bold'})

    plt.show()


def plot_window_expansion_lines(results_list, windows_ms,
                                cell_labels,
                                targets=('stim_mean', 'stim_std', 'stim_exp'),
                                target_labels=('stim mean', 'stim std', 'stim exp')):
    """
    Line plot of bootstrap R² across window sizes for multiple cells.

    Each cell gets a distinct line color + shaded 95% CI band. Significant windows
    are marked with a filled circle; non-significant with an open circle. Stars are
    placed at the top of the panel for significant windows per cell.

    Parameters
    ----------
    results_list : list of dict   – one results_by_window dict per cell
    windows_ms   : list of int    – window sizes (x-axis)
    cell_labels  : list of str    – label per cell for legend
    targets      : tuple of str
    target_labels: tuple of str
    """
    _FS_AX   = 28
    _FS_TK   = 24
    _FS_STR  = 32
    _LW_SP   = 2.5
    _LW_LINE = 3.5
    _COLORS  = ['#D55E00', '#0072B2', '#009E73', '#CC79A7']
    _ALPHAS  = [0.18, 0.18, 0.18, 0.18]

    n_targets = len(targets)
    n_cells   = len(results_list)
    xs        = np.arange(len(windows_ms))

    fig, axes = plt.subplots(1, n_targets, figsize=(8 * n_targets, 7))
    if n_targets == 1:
        axes = [axes]

    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.08, right=0.97, wspace=0.35)

    for ax, tgt, tgt_lbl in zip(axes, targets, target_labels):
        for ci, (results, label) in enumerate(zip(results_list, cell_labels)):
            col = _COLORS[ci % len(_COLORS)]
            r2s, lo, hi, sigs = [], [], [], []
            for wms in windows_ms:
                key = f'{wms}ms_{tgt}'
                if key not in results:
                    r2s.append(np.nan); lo.append(np.nan)
                    hi.append(np.nan); sigs.append(False)
                    continue
                res = results[key]
                r2s.append(res['r2_mean'])
                lo.append(res['r2_ci'][0])
                hi.append(res['r2_ci'][1])
                sigs.append(res.get('p_val_perm', 1.0) < 0.05)

            r2s = np.array(r2s, dtype=float)
            lo  = np.array(lo,  dtype=float)
            hi  = np.array(hi,  dtype=float)

            ax.fill_between(xs, lo, hi, color=col, alpha=_ALPHAS[ci], zorder=2)
            ax.plot(xs, r2s, color=col, lw=_LW_LINE, zorder=3,
                    label=label, solid_capstyle='round')

            # filled dot = sig, open dot = n.s.
            for xi, (r2v, sig) in enumerate(zip(r2s, sigs)):
                if np.isnan(r2v):
                    continue
                if sig:
                    ax.scatter(xs[xi], r2v, s=120, color=col,
                               zorder=4, edgecolors='black', linewidths=1.5)
                else:
                    ax.scatter(xs[xi], r2v, s=120, facecolors='white',
                               edgecolors=col, linewidths=2.0, zorder=4)

            # stars just above the CI upper bound
            star_offset = ci * 0.04 * (np.nanmax(hi) - np.nanmin(lo) + 1e-6)
            for xi, (hv, sig) in enumerate(zip(hi, sigs)):
                if sig and not np.isnan(hv):
                    ax.text(xs[xi], hv + star_offset + 0.008, '*',
                            ha='center', va='bottom', fontsize=_FS_STR,
                            fontweight='bold', color=col)

        ax.axhline(0, color='black', lw=1.8, ls='--', alpha=0.4, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(w) for w in windows_ms],
                           rotation=45, ha='right', fontsize=_FS_TK, fontweight='bold')
        ax.tick_params(axis='y', labelsize=_FS_TK, width=_LW_SP)
        ax.tick_params(axis='x', width=_LW_SP)
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight('bold')
        ax.set_xlabel('Pre-spike window (ms)', fontsize=_FS_AX, fontweight='bold', labelpad=10)
        ax.set_ylabel('Bootstrap R²', fontsize=_FS_AX, fontweight='bold', labelpad=10)
        ax.set_title(tgt_lbl, fontsize=_FS_AX, fontweight='bold', pad=16)
        ax.locator_params(axis='y', nbins=4)
        for spine in ax.spines.values():
            spine.set_linewidth(_LW_SP)
            spine.set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False,
               loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=n_cells, fontsize=_FS_TK,
               prop={'size': _FS_TK, 'weight': 'bold'})

    plt.show()


def plot_beta_weights_combined(results_window, window_ms=200):
    """Lollipop forest plot of significant bootstrap beta weights for all 3 stim targets.

    One row per significant (stim target × waveform feature) pair, sorted by |coef|.
    Colored by waveform feature (consistent with _FEATURE_COLOR_MAP). CI whiskers from
    bootstrap. Stars at the outer CI cap end.  Separate legend figure.
    Also outputs a bootstrapped R² bar chart as a separate figure.
    """
    _EXCLUDE     = {'sweep', 'spike_num', 'stim_type', 'pink_type'}
    _FEAT_DISPLAY = {
        'ramp_amp': 'Ramp Amp', 'inflection_time': 'Infl. Time',
        'inflection_amp': 'Infl. Amp', 'peak_amp': 'Peak Amp',
        'peak_width': 'Peak Width', 'peak_sharpness': 'Sharpness',
        'exp_lambda': 'Decay λ', 'exp_const': 'Decay Const', 'log_isi': 'Log ISI',
    }
    targets     = ['stim_mean', 'stim_std', 'stim_exp']
    tgt_labels  = {'stim_mean': 'Stim Mean', 'stim_std': 'Stim Std', 'stim_exp': 'Stim Exp'}

    sns.set_theme(style='ticks', rc={'axes.linewidth': 2.5})
    _FS_ROW, _FS_AX, _FS_STAR = 40, 46, 52

    res = {}
    for tgt in targets:
        key = f'{window_ms}ms_{tgt}'
        if key not in results_window:
            print(f"Key '{key}' not found in results_window — run window expansion first.")
            return
        res[tgt] = results_window[key]

    _DOT_HATCH = {'stim_mean': '',    'stim_std': '|', 'stim_exp': '/'}

    # collect significant pairs; group by feature, sort within by |coef|
    rows = []
    for tgt in targets:
        r = res[tgt]
        for feat, coef, ci_lo, ci_hi, pval in zip(
                r['feature_names'], r['coefficients'],
                r['ci_lower'], r['ci_upper'], r['p_values']):
            if feat in _EXCLUDE or pval >= 0.05:
                continue
            feat_lbl = _FEAT_DISPLAY.get(feat, feat)
            star = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
            rows.append(dict(tgt=tgt, feat=feat, feat_lbl=feat_lbl,
                             coef=float(coef), ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                             pval=float(pval), star=star,
                             abs_coef=abs(float(coef))))

    if not rows:
        print('No significant beta weights found.')
        return

    df = pd.DataFrame(rows)
    # sort: by feature importance (mean |coef| across targets), then fixed target order within
    feat_imp  = df.groupby('feat')['abs_coef'].mean().to_dict()
    tgt_order = {'stim_mean': 0, 'stim_std': 1, 'stim_exp': 2}
    df['feat_imp']  = df['feat'].map(feat_imp)
    df['tgt_order'] = df['tgt'].map(tgt_order)
    df = df.sort_values(['feat_imp', 'tgt_order'], ascending=[True, True]).reset_index(drop=True)
    n  = len(df)

    colors   = [_FEATURE_COLOR_MAP.get(f, '#888') for f in df['feat']]
    hatches  = [_DOT_HATCH[t] for t in df['tgt']]
    ci_lo_hw = (df['coef'] - df['ci_lo']).values
    ci_hi_hw = (df['ci_hi'] - df['coef']).values
    y        = np.arange(n, dtype=float)

    # add small gap between feature groups
    prev_feat = None
    gap_acc   = 0.0
    y_pos     = []
    for feat in df['feat']:
        if prev_feat is not None and feat != prev_feat:
            gap_acc += 0.5
        y_pos.append(len(y_pos) + gap_acc)
        prev_feat = feat
    y_pos = np.array(y_pos)

    # xlim / ylim — tight around actual CI extent
    all_ci_left  = float((df['coef'] - ci_lo_hw).min())
    all_ci_right = float((df['coef'] + ci_hi_hw).max())
    x_span   = all_ci_right - all_ci_left
    xlim_left  = all_ci_left  - x_span * 0.08
    xlim_right = all_ci_right + x_span * 0.28   # room for stars on right
    ylim_bot   = y_pos.min() - 0.8
    ylim_top   = y_pos.max() + 0.8

    # ellipse radius in data units — derived analytically (no canvas.draw())
    fig_w, fig_h  = 22, max(8, n * 1.15)
    ax_left, ax_right, ax_bot, ax_top = 0.25, 0.92, 0.15, 0.92
    ax_w_in = fig_w * (ax_right - ax_left)
    ax_h_in = fig_h * (ax_top   - ax_bot)
    dot_r_in = 0.30   # bigger dots
    r_x = dot_r_in / ax_w_in * (xlim_right - xlim_left)
    r_y = dot_r_in / ax_h_in * (ylim_top   - ylim_bot)

    matplotlib.rcParams['hatch.linewidth'] = 6.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.hlines(y_pos, 0, df['coef'], color=colors, linewidth=8.0, alpha=0.85, zorder=2)
    for yi, (coef, clo, chi) in enumerate(zip(df['coef'], ci_lo_hw, ci_hi_hw)):
        ax.errorbar(coef, y_pos[yi], xerr=[[clo], [chi]], fmt='none',
                    ecolor='black', elinewidth=2.5, capsize=12, capthick=2.5, zorder=4)

    # two-layer dots: layer1 hatch with white edgecolor → white hatch lines;
    # layer2 black border ring on top
    for yi, (coef, col, hatch) in enumerate(zip(df['coef'], colors, hatches)):
        ax.add_patch(matplotlib.patches.Ellipse(
            (coef, y_pos[yi]), width=2*r_x, height=2*r_y,
            facecolor=col, edgecolor='white', linewidth=6.0,
            hatch=hatch, zorder=5))
        ax.add_patch(matplotlib.patches.Ellipse(
            (coef, y_pos[yi]), width=2*r_x, height=2*r_y,
            facecolor='none', edgecolor='black', linewidth=2.5,
            zorder=6))

    ax.set_xlim(xlim_left, xlim_right)
    ax.set_ylim(ylim_bot, ylim_top)
    ax.axvline(0, color='black', lw=5.0, ls='--', alpha=0.7)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, prune='both'))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

    # faint horizontal separators between feature groups
    prev_feat2 = None
    for feat, yp in zip(df['feat'], y_pos):
        if prev_feat2 is not None and feat != prev_feat2:
            ax.axhline(yp - 0.75, color='#aaaaaa', lw=1.5, ls='-', alpha=0.5, zorder=1)
        prev_feat2 = feat

    # y-axis: feature name centered per group
    seen_feats, ytick_pos, ytick_lbl = [], [], []
    for feat in df['feat']:
        if feat not in seen_feats:
            seen_feats.append(feat)
    for feat in seen_feats:
        feat_ys = [yp for f, yp in zip(df['feat'], y_pos) if f == feat]
        ytick_pos.append(float(np.mean(feat_ys)))
        ytick_lbl.append(_FEAT_DISPLAY.get(feat, feat))

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lbl, fontsize=_FS_ROW, color='black')
    ax.set_xlabel('Bootstrap β', fontsize=_FS_AX,
                  fontweight='bold', color='black', labelpad=14)
    ax.tick_params(axis='x', labelsize=_FS_ROW, colors='black', pad=10,
                   width=3.0, length=10)
    ax.tick_params(axis='y', left=False)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.spines['left'].set_linewidth(4.5)
    sns.despine(ax=ax)
    fig.subplots_adjust(left=ax_left, right=ax_right, bottom=ax_bot, top=ax_top)
    plt.show()

    # legend: stim target → dot style
    tgt_label_map = {'stim_mean': 'Stim mean', 'stim_std': 'Stim std', 'stim_exp': 'Stim exp'}
    handles_leg = []
    for tgt in targets:
        h = _DOT_HATCH[tgt]
        patch = matplotlib.patches.Patch(facecolor='white', edgecolor='black',
                                         linewidth=2, hatch=h,
                                         label=tgt_label_map[tgt])
        handles_leg.append(patch)
    fig_leg, ax_leg = plt.subplots(figsize=(4, 2.5))
    ax_leg.axis('off')
    ax_leg.legend(handles=handles_leg, fontsize=_FS_ROW, frameon=False, loc='center')
    plt.tight_layout()
    plt.show()

    # ── Bootstrapped R² bar chart (original style) ───────────────────────────
    r2_means = [res[tgt]['r2_mean']   for tgt in targets]
    r2_lo    = [res[tgt]['r2_ci'][0]  for tgt in targets]
    r2_hi    = [res[tgt]['r2_ci'][1]  for tgt in targets]
    xerr_lo2 = [m - lo for m, lo in zip(r2_means, r2_lo)]
    xerr_hi2 = [hi - m for m, hi in zip(r2_means, r2_hi)]
    r2_hatch = ['', '|', '/']
    xs2      = np.arange(len(targets))

    fig2, ax2 = plt.subplots(figsize=(12, 9), constrained_layout=True)
    for i, (tgt, h2) in enumerate(zip(targets, r2_hatch)):
        ax2.bar(xs2[i], r2_means[i], color='white', alpha=1.0,
                edgecolor='#1a1a1a', linewidth=5, width=0.62,
                yerr=[[xerr_lo2[i]], [xerr_hi2[i]]],
                error_kw=dict(ecolor='#1a1a1a', lw=5, capsize=16, capthick=5))
        if h2 == '|':
            bw     = 0.62
            x_left = xs2[i] - bw / 2
            bar_top = r2_means[i]
            clip_v = matplotlib.patches.Rectangle((x_left, 0), bw, bar_top,
                                                   transform=ax2.transData)
            ax2.add_patch(clip_v); clip_v.set_visible(False)
            lc = ax2.vlines(np.linspace(x_left, xs2[i] + bw / 2, 5)[1:-1],
                            0, bar_top, colors='#1a1a1a', linewidth=5, zorder=4)
            lc.set_clip_path(clip_v)
        elif h2 == '/':
            bw       = 0.62
            x_left_b = xs2[i] - bw / 2
            x_right_b = xs2[i] + bw / 2
            bar_top  = r2_means[i]
            clip_d   = matplotlib.patches.Rectangle((x_left_b, 0), bw, bar_top,
                                                     transform=ax2.transData)
            ax2.add_patch(clip_d); clip_d.set_visible(False)
            N_diag = 4
            for delta in np.linspace(-(N_diag-1)/(N_diag+1),
                                      (N_diag-1)/(N_diag+1), N_diag) * bw:
                _xs  = x_left_b + delta
                t_lo = max(0.0, (x_left_b  - _xs) / bw)
                t_hi = min(1.0, (x_right_b - _xs) / bw)
                if t_lo >= t_hi:
                    continue
                line, = ax2.plot([_xs + t_lo*bw, _xs + t_hi*bw],
                                 [t_lo*bar_top,   t_hi*bar_top],
                                 color='#1a1a1a', lw=5, solid_capstyle='butt', zorder=4)
                line.set_clip_path(clip_d)
        p_perm = res[tgt].get('p_val_perm', None)
        if p_perm is not None:
            star = '***' if p_perm < 0.001 else '**' if p_perm < 0.01 else '*' if p_perm < 0.05 else ''
            if star:
                ax2.text(xs2[i], r2_hi[i] + 0.005, star, ha='center', va='bottom',
                         fontsize=100, color='black', fontweight='bold')

    ax2.set_xticks(xs2)
    ax2.set_xticklabels(['mean', 'stdev', 'exp'], fontsize=56, fontweight='bold')
    ax2.set_ylabel('Bootstrapped R²', fontsize=60, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=52, width=3, length=8)
    ax2.tick_params(axis='x', width=0, length=0)
    ax2.locator_params(axis='y', nbins=4)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    for lbl in ax2.get_yticklabels():
        lbl.set_fontweight('bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(5)
    ax2.spines['bottom'].set_linewidth(5)
    plt.show()


def plot_fit_quality_distributions(sp):
    """Histograms of R² for the ramp-amplitude and exp-decay fits across all spikes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, vals, xlabel, color in zip(
        axes,
        [sp.r_squared_ramp, sp.r_squared_exp],
        ['Ramp fit R²', 'Exp decay fit R²'],
        ['C4', 'C6'],
    ):
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        ax.hist(v, bins=40, color=color, alpha=0.85, edgecolor='none')
        med = np.median(v)
        ax.axvline(med, color='k', lw=2.5, ls='--', label=f'median = {med:.3f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.legend(frameon=False)
        ax.locator_params(nbins=4)
    plt.show()


def plot_feature_histograms(df_features):
    """Histogram of every spike waveform feature column in df_features."""
    _skip = {'sweep', 'stim_type', 'pink_type', 'spike_num',
             'stim_exp', 'stim_mean', 'stim_std', 'r_squared_exp', 'r_squared_ramp'}
    feat_cols = [c for c in df_features.columns if c not in _skip]
    n = len(feat_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5),
                             constrained_layout=True)
    axes = np.array(axes).flatten()
    for ax, col in zip(axes, feat_cols):
        vals = df_features[col].dropna().values
        color = _FEATURE_COLOR_MAP.get(col, 'steelblue')
        ax.hist(vals, bins=40, color=color, alpha=0.85, edgecolor='none')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.locator_params(nbins=3)
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.show()


def plot_feature_correlation_matrix(df_features, title=''):
    _FS_AX = 26
    _FS_TK = 20
    _LW_SP = 2.0
    _SKIP  = {'sweep', 'stim_type', 'pink_type', 'spike_num',
              'stim_exp', 'stim_mean', 'stim_std', 'r_squared_exp',
              'r_squared_ramp', 'log_isi'}
    _LABELS = {
        'peak_amp':       'Peak amp',
        'peak_width':     'Peak width',
        'peak_sharpness': 'Peak sharpness',
        'exp_lambda':     'Exp λ',
        'exp_const':      'Exp const',
        'inflection_time':'Inflection time',
        'inflection_amp': 'Inflection amp',
        'ramp_amp':       'Ramp amp',
    }

    feat_cols = [c for c in df_features.columns if c not in _SKIP]
    corr = df_features[feat_cols].corr(method='pearson')
    labels = [_LABELS.get(c, c) for c in feat_cols]

    n = len(feat_cols)
    fig, ax = plt.subplots(figsize=(n * 1.1 + 1.5, n * 1.1 + 1.0), constrained_layout=True)

    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=_FS_TK, fontweight='bold')
    ax.set_yticklabels(labels, fontsize=_FS_TK, fontweight='bold')

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            txt_col = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=_FS_TK - 4, fontweight='bold', color=txt_col)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label('Pearson r', fontsize=_FS_AX - 2, fontweight='bold')
    cbar.ax.tick_params(labelsize=_FS_TK - 2)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight('bold')

    if title:
        ax.set_title(title, fontsize=_FS_AX, fontweight='bold', pad=14)

    for spine in ax.spines.values():
        spine.set_linewidth(_LW_SP)
    plt.show()

