import matplotlib.pyplot as plt
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
    'exp_lambda', 'exp_const', 'log_isi',
]
_STIM_FEATURES = ['stim_mean', 'stim_std', 'stim_exp']

# Bold palette — one colour per stim feature, reused across subplots
_SCATTER_COLORS = {'stim_mean': '#E07B54', 'stim_std': '#5B8DB8', 'stim_exp': '#72B26C'}


def plot_top_correlations_by_window(df_w, window_ms, n_top=6):
    """2×3 grid of the top-N spike-feature × stim-feature correlations for a given window.

    Pairs are ranked by |Pearson r|. Style is bold/cartoony with minimal tick clutter.
    """
    pairs = []
    for sf in _SPIKE_FEATURES:
        for stf in _STIM_FEATURES:
            if sf not in df_w.columns or stf not in df_w.columns:
                continue
            valid = df_w[[sf, stf]].dropna()
            if len(valid) < 10:
                continue
            r, p = pearsonr(valid[sf], valid[stf])
            pairs.append((abs(r), r, p, sf, stf, valid))

    pairs.sort(key=lambda x: x[0], reverse=True)
    top = pairs[:n_top]

    ncols = 3
    nrows = int(np.ceil(n_top / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(11 * ncols, 9 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for ax, (abs_r, r, p, sf, stf, valid) in zip(axes, top):
        color = _FEATURE_COLOR_MAP.get(sf, '#888')
        ax.scatter(valid[sf], valid[stf],
                   s=100, alpha=0.65, color=color, linewidths=0)

        # regression line
        m, b = np.polyfit(valid[sf], valid[stf], 1)
        xs = np.linspace(valid[sf].min(), valid[sf].max(), 200)
        ax.plot(xs, m * xs + b, color='#1a1a1a', lw=4, zorder=3)

        # r / p annotation — bottom-right, no title needed
        p_str = 'p<0.001' if p < 0.001 else f'p={p:.3f}'
        ax.text(0.96, 0.05, f'r = {r:.2f}\n{p_str}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=28, fontweight='bold', color='#1a1a1a')

        ax.set_xlabel(sf.replace('_', ' '), fontsize=34, fontweight='bold')
        ax.set_ylabel(f'{window_ms} ms {stf.replace("_", " ")}', fontsize=34, fontweight='bold')

        ax.tick_params(axis='both', labelsize=28, width=2.5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['bottom'].set_linewidth(2.5)

        # 3 ticks per axis — no clutter
        ax.locator_params(nbins=3)

    # hide unused axes
    for ax in axes[len(top):]:
        ax.set_visible(False)

    plt.show()

def plot_avg_waveform_by_stim_type(all_constant_spks, all_ramp_spks, all_pink_spks):
    """Mean ± SD spike waveform derivative for each stimulus type (constant / ramp / pink)."""
    mean_constant_spks = np.mean(all_constant_spks, axis=0)
    std_constant_spks = np.std(all_constant_spks, axis=0)
    
    mean_ramp_spks = np.mean(all_ramp_spks, axis=0)
    std_ramp_spks = np.std(all_ramp_spks, axis=0)
    
    mean_pink_spks = np.mean(all_pink_spks, axis=0)
    std_pink_spks = np.std(all_pink_spks, axis=0)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(mean_constant_spks, label='Mean Constant Spikes', color='green', linewidth=4)
    plt.fill_between(range(len(mean_constant_spks)), mean_constant_spks - std_constant_spks, mean_constant_spks + std_constant_spks, color='green', alpha=0.3)
    
    plt.plot(mean_ramp_spks, label='Mean Ramp Spikes', color='purple', linewidth=4)
    plt.fill_between(range(len(mean_ramp_spks)), mean_ramp_spks - std_ramp_spks, mean_ramp_spks + std_ramp_spks, color='purple', alpha=0.3)
    
    plt.plot(mean_pink_spks, label='Mean Pink Spikes', color='hotpink', linewidth=4)
    plt.fill_between(range(len(mean_pink_spks)), mean_pink_spks - std_pink_spks, mean_pink_spks + std_pink_spks, color='lightpink', alpha=0.3)
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage')
    plt.xlim(1600, 2400)
    plt.show()



def plot_confusion_matrix(best_model, X_test, y_test):
    """Confusion matrix for a trained classifier on test data."""
    from sklearn.metrics import confusion_matrix
    cm      = confusion_matrix(y_test, best_model.predict(X_test))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)

    fig, ax = plt.subplots(figsize=(11, 5))
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
                    raise ValueError(
                        f"y size {len(y)} != y_pred_cv size {len(yhat)} for key '{key}'. "
                        "Set FORCE_RERUN=True to regenerate pickles."
                    )
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
def plot_bootstrap_histograms(bootstrapped_results, model_names):
    n = len(bootstrapped_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, accs, model_name in zip(axes, bootstrapped_results, model_names):
        accs_mean = np.mean(accs)
        accs_ci   = np.percentile(accs, [2.5, 97.5])
        sns.histplot(accs, kde=True, bins=30, ax=ax)
        ax.axvline(accs_mean,   color='red',  ls='--', lw=2, label=f'Mean: {accs_mean:.3f}')
        ax.axvline(accs_ci[0],  color='gray', ls=':',  lw=1.5,
                   label=f'95% CI [{accs_ci[0]:.3f},{accs_ci[1]:.3f}]')
        ax.axvline(accs_ci[1],  color='gray', ls=':',  lw=1.5)
        ax.set_xlabel(f'Accuracy  ({model_name})')
        ax.set_ylabel('Frequency')
        ax.legend(frameon=False)
    plt.show()


def plot_window_expansion(results_by_window, windows_ms,
                          targets=('stim_mean', 'stim_std', 'stim_exp'),
                          target_labels=('stim mean', 'stim std', 'stim exp'),
                          title='',
                          shuffle_results=None):
    """
    Compare ridge regression R² (bootstrap mean + 95 % CI) across pre-inflection
    window sizes for one or more stimulus targets.

    Parameters
    ----------
    results_by_window : dict
        Keyed as '{W}ms_{target}', e.g. '5ms_stim_mean'.
    windows_ms      : list of int   – window sizes tested (x-axis)
    targets         : tuple of str  – target names (separate subplots)
    target_labels   : tuple of str  – display labels for targets
    title           : str           – figure suptitle suffix
    shuffle_results : dict, optional
        Same key scheme as results_by_window but for shuffled-Y regressions
        (keyed as '{W}ms_{target}'). Shown as lighter hatched bars paired with
        each real bar to visualise the null R² at every window size.
    """
    bar_w     = 0.35
    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets,
                             figsize=(5.5 * n_targets, 5.5),
                             constrained_layout=True)
    if n_targets == 1:
        axes = [axes]

    for col_i, (ax, tgt, tgt_lbl) in enumerate(zip(axes, targets, target_labels)):
        r2s, lo, hi, sigs = [], [], [], []
        for wms in windows_ms:
            key = f'{wms}ms_{tgt}'
            if key not in results_by_window:
                r2s.append(np.nan); lo.append(np.nan); hi.append(np.nan); sigs.append(False)
                continue
            res = results_by_window[key]
            r2s.append(res['r2_mean'])
            lo.append(res['r2_ci'][0])
            hi.append(res['r2_ci'][1])
            sigs.append(res.get('p_val_perm', 1.0) < 0.05)

        r2s = np.array(r2s, dtype=float)
        lo  = np.array(lo,  dtype=float)
        hi  = np.array(hi,  dtype=float)

        if shuffle_results is not None:
            xs_real  = np.arange(len(windows_ms)) - bar_w / 2
            xs_shuf  = np.arange(len(windows_ms)) + bar_w / 2
            xs_ticks = np.arange(len(windows_ms))
        else:
            xs_real  = np.arange(len(windows_ms))
            xs_ticks = xs_real

        colors = ['#D55E00' if s else '#888888' for s in sigs]
        ax.bar(xs_real, r2s, color=colors, alpha=0.85, width=bar_w)
        ax.errorbar(xs_real, r2s,
                    yerr=[r2s - lo, hi - r2s],
                    fmt='none', color='k', capsize=4, lw=1.5)

        if shuffle_results is not None:
            sr2s, slo, shi = [], [], []
            for wms in windows_ms:
                skey = f'{wms}ms_{tgt}'
                if skey not in shuffle_results:
                    sr2s.append(np.nan); slo.append(np.nan); shi.append(np.nan)
                    continue
                sr = shuffle_results[skey]
                sr2s.append(sr['r2_mean'])
                slo.append(sr['r2_ci'][0])
                shi.append(sr['r2_ci'][1])
            sr2s = np.array(sr2s, dtype=float)
            slo  = np.array(slo,  dtype=float)
            shi  = np.array(shi,  dtype=float)
            ax.bar(xs_shuf, sr2s, color='#56B4E9', alpha=0.6, width=bar_w,
                   hatch='//', edgecolor='white', linewidth=0.4)
            ax.errorbar(xs_shuf, sr2s,
                        yerr=[sr2s - slo, shi - sr2s],
                        fmt='none', color='#56B4E9', capsize=4, lw=1.5)

        ax.axhline(0, color='k', lw=1, ls='--', alpha=0.5)
        ax.set_xticks(xs_ticks)
        ax.set_xticklabels([str(w) for w in windows_ms], rotation=45, ha='right')
        ax.set_xlabel('Window (ms)')
        # target label on y-axis instead of panel title
        ax.set_ylabel(f'Bootstrap R²\n({tgt_lbl})')
        ax.locator_params(axis='y', nbins=4)

        for xi, (r2, sig) in enumerate(zip(r2s, sigs)):
            if sig:
                ax.text(xs_real[xi], hi[xi] + 0.01, '*',
                        ha='center', va='bottom', color='#D55E00')

    # legend outside axes in top-right of figure
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#D55E00', alpha=0.85, label='p < 0.05 (perm.)'),
                       Patch(facecolor='#888888', alpha=0.85, label='n.s.')]
    if shuffle_results is not None:
        legend_elements.append(
            Patch(facecolor='#56B4E9', alpha=0.6, hatch='//', label='shuffle ctrl')
        )
    fig.legend(handles=legend_elements, frameon=False,
               loc='lower center', bbox_to_anchor=(0.5, 1.01),
               ncol=len(legend_elements))

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

