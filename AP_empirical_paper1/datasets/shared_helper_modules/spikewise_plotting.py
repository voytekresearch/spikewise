import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection
import os

plt.rcParams['font.family'] = 'Helvetica Neue'


def plot_corr_heatmap_only_spk(
    df_features,
    spike_features,
    calculate_corr=False,
    corr_threshold=0.1,
    title: str = None,
    figsize=(14, 11),
):
    """Lower-triangle correlation heatmap with cartoony styling: big stars, white text on dark cells."""
    # ── Built-in label abbreviations (no underscores) ─────────────────────────
    _LABEL = {
        'ramp_amp':       'ramp amp',
        'inflection_time':'inflec time',
        'inflection_amp': 'inflec amp',
        'peak_amp':       'peak amp',
        'peak_width':     'peak width',
        'peak_sharpness': 'peak sharp',
        'exp_lambda':     'exp lambda',
        'exp_const':      'exp const',
        'log_isi':        'log ISI',
        'isi':            'log ISI',
    }
    def _short(f):
        return _LABEL.get(f, f.replace('_', ' '))

    def _y_label(f):
        s = _short(f)
        parts = s.split(' ')
        # split two-word labels longer than 8 chars onto two lines
        if len(parts) == 2:
            return parts[0] + '\n' + parts[1]
        return s

    df_selected = df_features[spike_features]
    df_cleaned = df_selected.replace([np.inf, -np.inf], np.nan).dropna()
    if df_cleaned.empty:
        raise ValueError("No valid data after cleaning.")

    rho = df_cleaned.corr() if calculate_corr else df_features[spike_features].corr()

    n = len(spike_features)
    pval = pd.DataFrame(np.ones_like(rho), index=spike_features, columns=spike_features)
    qval = pd.DataFrame(np.ones_like(rho), index=spike_features, columns=spike_features)
    pairs = [(row, col) for ci, col in enumerate(spike_features)
             for row in spike_features[ci + 1:]]
    raw_p = []
    for row, col in pairs:
        _, p = pearsonr(df_cleaned[row], df_cleaned[col])
        pval.at[row, col] = p
        raw_p.append(p)
    if raw_p:
        _, q_fdr = fdrcorrection(raw_p, alpha=0.05, method='indep')
        for (row, col), q in zip(pairs, q_fdr):
            qval.at[row, col] = q
    mask    = np.triu(np.ones_like(rho, dtype=bool))
    # x: one line, angled  |  y: two lines where label is long
    xlabels = [_short(f) for f in spike_features[:-1]] + ['']
    ylabels = [''] + [_y_label(f) for f in spike_features[1:]]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(rho, mask=mask, annot=False, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0, ax=ax,
                xticklabels=xlabels, yticklabels=ylabels)

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['-1', '0', '1'])
    cbar.ax.tick_params(labelsize=22, width=2.5, length=6)
    for spine in cbar.ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(2.5); spine.set_color('#1a1a1a')
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold'); label.set_fontfamily('Helvetica Neue')

    # L-shaped border only (left + bottom) — avoids rectangle around empty upper triangle
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(2.5)
        ax.spines[spine].set_color('#1a1a1a')

    _cmap_obj = plt.cm.RdBu_r
    _norm_obj = mcolors.Normalize(vmin=-1, vmax=1)
    cell_h_in = figsize[1] / n
    star_fs = max(14, int(cell_h_in * 72 * 0.45))
    r_fs    = max(10, int(cell_h_in * 72 * 0.26))
    star_y = 0.32
    num_y  = 0.68

    for i in range(n):
        for j in range(n):
            if i > j:
                r_val = rho.iloc[i, j]
                p     = pval.iloc[i, j]
                sig   = p < 0.05 and abs(r_val) >= corr_threshold
                rgba      = _cmap_obj(_norm_obj(r_val))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = 'white' if luminance < 0.5 else 'black'

                ax.text(j + 0.5, i + num_y, f"{r_val:.2f}",
                        ha='center', va='center', fontsize=r_fs, color=txt_color,
                        fontweight='bold' if p < 0.05 else 'normal',
                        fontfamily='Helvetica Neue')
                if sig:
                    star = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                    ax.text(j + 0.5, i + star_y, star,
                            ha='center', va='center', fontsize=star_fs,
                            fontweight='bold', color=txt_color,
                            fontfamily='Helvetica Neue')

    # x: angled labels, larger font  |  y: two-line labels, slightly smaller to fit in cell
    ax.tick_params(axis='x', which='major', labelsize=32, width=2.5)
    ax.tick_params(axis='y', which='major', labelsize=28, width=2.5)
    fig.canvas.draw()
    for label in ax.get_xticklabels():
        label.set_fontweight('bold'); label.set_fontfamily('Helvetica Neue')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold'); label.set_fontfamily('Helvetica Neue')
        label.set_multialignment('center')
    if title:
        ax.set_title(title, fontsize=22, fontweight='bold', fontfamily='Helvetica Neue')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def display_feature_table(df_features, nrows=2, save_path=None, blend=0.20):
    """Display a styled spike feature DataFrame and optionally save as PNG.

    Args:
        df_features: DataFrame from sp.df_features (before QC filtering so r2 cols are present).
        nrows: Number of spike rows to show.
        save_path: If given, saves a PNG to this path (uses matplotlib, no extra deps).
        blend: Color saturation for column header tints (0=white, 1=full color). Default 0.20.
    """
    from IPython.display import display, HTML

    _feat_colors = {
        'ramp_amp':        'C4',
        'inflection_time': 'C3',
        'inflection_amp':  'C3',
        'peak_amp':        'C5',
        'peak_width':      'C5',
        'peak_sharpness':  'C5',
        'exp_lambda':      'C6',
        'exp_const':       'C6',
        'isi':             'C7',
        'log_isi':         'C7',
        'r_squared_ramp':  'C4',
        'r_squared_exp':   'C6',
    }
    # 'log_isi' is listed after 'isi' so whichever the df has gets picked up
    _col_order = ['ramp_amp', 'inflection_time', 'inflection_amp',
                  'peak_amp', 'peak_width', 'peak_sharpness',
                  'exp_lambda', 'exp_const', 'isi', 'log_isi',
                  'r_squared_ramp', 'r_squared_exp']
    _col_labels_html = {
        'ramp_amp':        'ramp amp',
        'inflection_time': 'inflection<br>time',
        'inflection_amp':  'inflection<br>amp',
        'peak_amp':        'peak<br>amp',
        'peak_width':      'peak<br>width',
        'peak_sharpness':  'peak<br>sharpness',
        'exp_lambda':      'exp<br>lambda',
        'exp_const':       'exp<br>const',
        'isi':             'log<br>isi',
        'log_isi':         'log<br>isi',
        'r_squared_ramp':  'r2 ramp<br>fit',
        'r_squared_exp':   'r2 decay<br>fit',
    }

    def _faint(c):
        rgb = np.array(mcolors.to_rgb(c))
        blended = rgb * blend + np.ones(3) * (1 - blend)
        r, g, b = [int(x * 255) for x in blended]
        return f'rgb({r},{g},{b})'

    def _faint_rgb(c):
        rgb = np.array(mcolors.to_rgb(c))
        return rgb * blend + np.ones(3) * (1 - blend)

    _avail = [c for c in _col_order if c in df_features.columns]
    _df = df_features[_avail].head(nrows).copy()
    if 'isi' in _df.columns:
        _df['isi'] = np.log10(_df['isi'])
    # log_isi is already logged — no transform needed
    _df = _df.round(2).rename(columns=_col_labels_html)
    # use sequential 0,1,2... as spike id regardless of original index
    _df.index = range(len(_df))
    _df.index.name = 'spike id'
    _df = _df.reset_index()

    # ── HTML display ──────────────────────────────────────────────────────────
    import re as _re
    _html = _df.to_html(escape=False, index=False)
    # Strip pandas' default class/border attrs so Jupyter's .dataframe CSS doesn't fight ours
    _html = _re.sub(r'<table[^>]*>', '<table class="spk-df">', _html)

    _col_css = '\n'.join(
        f'table.spk-df thead th:nth-child({i+2}) {{ background-color: {_faint(_feat_colors[c])} !important; }}'
        for i, c in enumerate(_avail)
    )
    display(HTML(f"""<style>
table.spk-df {{
    border-collapse: collapse !important;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
}}
table.spk-df th {{
    font-size: 26px !important;
    font-weight: bold !important;
    border: 2.5px solid #1a1a1a !important;
    padding: 7px 16px !important;
    text-align: center !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    line-height: 1.3 !important;
    vertical-align: middle !important;
    white-space: nowrap !important;
    min-width: 90px !important;
}}
table.spk-df td {{
    font-size: 24px !important;
    font-weight: normal !important;
    border: 1.5px solid #1a1a1a !important;
    padding: 6px 16px !important;
    text-align: center !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    vertical-align: middle !important;
    white-space: nowrap !important;
}}
{_col_css}
</style>{_html}"""))

    # ── PNG save ──────────────────────────────────────────────────────────────
    # Use Figure + FigureCanvasAgg directly (never touches plt) so %matplotlib
    # inline does not capture or display this figure in the notebook output.
    if save_path is not None:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        _plain_labels = {c: _col_labels_html.get(c, c).replace('<br>', '\n') for c in _avail}
        _df_png = df_features[_avail].head(nrows).copy()
        if 'isi' in _df_png.columns:
            _df_png['isi'] = np.log10(_df_png['isi'])
        _df_png = _df_png.round(2).rename(columns=_plain_labels)
        _df_png.index = range(len(_df_png))
        _df_png.index.name = 'spike\nid'
        _df_png = _df_png.reset_index()

        n_cols = len(_df_png.columns)
        col_w, row_h = 2.1, 1.0
        # spike id only ever holds a single digit, so give it less width than
        # the feature columns and spend the freed-up space on bigger text
        _col_weights = [0.6] + [1.0] * (n_cols - 1)
        _weight_sum = sum(_col_weights)
        _col_widths = [w / _weight_sum for w in _col_weights]
        fig = Figure(figsize=(col_w * _weight_sum, (nrows + 1) * row_h + 0.6),
                     facecolor='white')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.axis('off')

        tbl = ax.table(
            cellText=_df_png.values.tolist(),
            colLabels=list(_df_png.columns),
            colWidths=_col_widths,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1],
        )
        tbl.auto_set_font_size(False)

        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor('#1a1a1a')
            if row == 0:
                cell.set_linewidth(2.5)
                cell.set_text_props(fontweight='bold', fontsize=26,
                                    fontfamily='Helvetica Neue')
                feat = _avail[col - 1] if (col > 0 and col - 1 < len(_avail)) else None
                cell.set_facecolor(_faint_rgb(_feat_colors[feat]) if feat and feat in _feat_colors else np.ones(3))
            else:
                cell.set_linewidth(1.5)
                cell.set_facecolor(np.ones(3))
                cell.set_text_props(fontsize=24, fontfamily='Helvetica Neue')

        fig.savefig(save_path, dpi=180, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved to {save_path}')


def plot_corr_heatmap(
    df_features,
    spike_features,
    lfp_features,
    calculate_corr=False,
    type_heatmap="half",
    show_sig=True,
    cmap="coolwarm",
    corr_threshold=0.1,  # Correlation threshold for plotting stars
    star_offset=0.2,  # Adjust to move the stars above the numbers
    title: str = "Correlation Heatmap"
):
    """
    Plots correlation heatmap of spike and LFP features with optional significance stars.

    Args:
        df_features: pandas DataFrame containing spike and LFP features.
        spike_features: List of spike feature column names.
        lfp_features: List of LFP feature column names.
        calculate_corr: If True, compute correlations from scratch. Otherwise, assume it's precomputed.
        type_heatmap: "half" or "full".
        show_sig: Whether to overlay significance stars.
        cmap: Color map for heatmap.
        corr_threshold: Minimum correlation value to display significance stars (default: 0.1).
        star_offset: Adjust this to move the stars vertically (default: 0.2).
    """

    # Select only relevant features
    df_selected = df_features[spike_features + lfp_features]

    # Clean data by removing NaN and Inf values
    df_cleaned = df_selected.replace([np.inf, -np.inf], np.nan).dropna()
    if df_cleaned.empty:
        raise ValueError("After cleaning, no data remains. Check for excessive missing values.")

    # Always compute correlation from the cleaned DataFrame to avoid NaN/Inf contamination
    correlation_matrix = df_cleaned.corr()

    # Extract the correct matrix with spikes on Y and LFPs on X
    rho = correlation_matrix.loc[spike_features, lfp_features]

    # Compute p-values using Pearson correlation (always uses cleaned data)
    pval = pd.DataFrame(np.zeros_like(rho), index=spike_features, columns=lfp_features)
    for row in spike_features:
        for col in lfp_features:
            r, p = pearsonr(df_cleaned[row], df_cleaned[col])
            pval.at[row, col] = p

    # Convert p-values to stars (show only if correlation > threshold)
    def p_to_star(p, r):
        if abs(r) < corr_threshold:
            return ""  # No stars for low correlations
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        else: return ''
    
    # Initialize the stars DataFrame with object (string) dtype to avoid warnings
    stars = pd.DataFrame("", index=rho.index, columns=rho.columns, dtype="object")

    # Apply stars using the significance function
    for i in range(len(spike_features)):
        for j in range(len(lfp_features)):
            stars.iloc[i, j] = p_to_star(pval.iloc[i, j], rho.iloc[i, j])

    # Handle half matrix
    if type_heatmap == "half" and rho.shape[0] == rho.shape[1]:
        mask = np.triu(np.ones_like(rho, dtype=bool))
        square = True
    else:
        mask = None
        square = False

    # Plot correlation values
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        rho,
        mask=mask,
        annot=rho.round(2),
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        center=0,
        square=square,
        xticklabels=lfp_features,
        yticklabels=spike_features,
        annot_kws={"size": 8, "color": "black"}
    )

    # ✅ Plot significance stars above the correlation numbers
    if show_sig:
        for i in range(len(spike_features)):
            for j in range(len(lfp_features)):
                if stars.iloc[i, j] != "":
                    plt.text(j + 0.5, i + 0.35 - star_offset, stars.iloc[i, j],
                             ha='center', va='center', color='black', fontsize=10)

    # Adjust axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.title(title)
    plt.tight_layout()
    plt.show()

def save_foof_fit_examples(foof_by_config, output_dir="foof_fit_plots", n_examples=3):
    os.makedirs(output_dir, exist_ok=True)
    for config_id, foof_list in foof_by_config.items():
        for i, fm in enumerate(foof_list[:n_examples]):
            fig = fm.plot(plot_peaks="shade", add_legend=True)
            fname = f"{config_id}_example_{i}.png"
            path = os.path.join(output_dir, fname)
            plt.savefig(path)
            plt.close()


def plot_fits_near_global_mean_and_outliers(
    results_df,
    foof_by_config,
    metric: str = "aperiodic_exponent",
    n_mean: int = 3,
    n_outliers: int = 3
):
    """
    Selects a config whose mean metric value is closest to the global mean,
    then plots:
        - FOOOF fits closest to that config's own mean
        - Outlier fits (lowest and highest) from that same config

    Parameters:
    - results_df: DataFrame from sensitivity_analysis
    - foof_by_config: Dict mapping config_id to list of FOOOF objects
    - metric: The metric to base plots on (e.g. "aperiodic_exponent")
    - n_mean: Number of fits near the mean to plot
    - n_outliers: Number of low/high outliers to plot
    """
    import matplotlib.pyplot as plt

    if metric not in results_df.columns:
        print(f"Metric '{metric}' not found in results_df.")
        return

    global_mean = results_df[metric].mean()
    config_means = results_df.groupby("config_id")[metric].mean()
    chosen_config_id = (config_means - global_mean).abs().sort_values().index[0]

    print(f"\nSelected config_id: {chosen_config_id} (mean {metric} = {config_means[chosen_config_id]:.2f}, global = {global_mean:.2f})")

    subset = results_df[results_df["config_id"] == chosen_config_id].copy().reset_index(drop=True)
    foofs = foof_by_config[chosen_config_id]

    if len(subset) != len(foofs):
        print("Warning: Number of FOOOF objects does not match subset rows.")
        return

    config_mean = subset[metric].mean()
    subset["abs_diff_from_mean"] = (subset[metric] - config_mean).abs()

    
    closest_idx = list(subset.sort_values("abs_diff_from_mean").iloc[:n_mean].index.values)

    print(f"\nPlotting {n_mean} fits closest to config mean ({metric} = {config_mean:.2f}):")
    for i in closest_idx:
        fm = foofs[i]
        val = fm.get_params("aperiodic_params", "exponent")
        print(f"Fit #{i}: DataFrame {metric} = {subset.iloc[i][metric]:.2f}, FOOOF exponent = {val:.2f}")
        fm.plot(plot_peaks="shade", add_legend=True)
        plt.title(f"{metric} ≈ {val:.2f} (fit #{i})")
        plt.show()

    # Outlier indices (also aligned via iloc)
    outlier_idx = list(pd.concat([
        subset.sort_values(metric).iloc[:n_outliers],
        subset.sort_values(metric).iloc[-n_outliers:]
    ]).index.values)

    print(f"\nPlotting {2 * n_outliers} outlier fits based on {metric}:")
    for i in outlier_idx:
        fm = foofs[i]
        val = fm.get_params("aperiodic_params", "exponent")
        print(f"Outlier fit #{i}: DataFrame {metric} = {subset.iloc[i][metric]:.2f}, FOOOF exponent = {val:.2f}")
        fm.plot(plot_peaks="shade", add_legend=True)
        plt.title(f"Outlier {metric} = {val:.2f} (fit #{i})")
        plt.show()


