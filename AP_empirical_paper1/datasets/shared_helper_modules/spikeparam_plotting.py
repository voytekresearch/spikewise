import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from scipy.stats import pearsonr
import os 


def plot_corr_heatmap_only_spk(
    df_features,
    spike_features,
    calculate_corr=False,
    corr_threshold=0.1,
    title: str = None,
    drop_y_label=None,
    drop_x_label=None,
):
    """
    Heatmap with larger correlation numbers, stars, and axis label font sizes.
    """

    # Clean data
    df_selected = df_features[spike_features]
    df_cleaned = df_selected.replace([np.inf, -np.inf], np.nan).dropna()
    if df_cleaned.empty:
        raise ValueError("No valid data after cleaning.")

    # Correlation and p-values
    if calculate_corr:
        rho = df_cleaned.corr()
    else:
        rho = df_features[spike_features].corr()

    pval = pd.DataFrame(np.zeros_like(rho), index=spike_features, columns=spike_features)
    for row in spike_features:
        for col in spike_features:
            r, p = pearsonr(df_cleaned[row], df_cleaned[col])
            pval.at[row, col] = p

    def p_to_star(p, r):
        if abs(r) < corr_threshold:
            return ""
        if p < 0.001: return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        else: return ''

    stars = pd.DataFrame("", index=rho.index, columns=rho.columns, dtype="object")
    for i in range(len(spike_features)):
        for j in range(len(spike_features)):
            stars.iloc[i, j] = p_to_star(pval.iloc[i, j], rho.iloc[i, j])

    # Trim labels
    xticks = spike_features.copy()
    yticks = spike_features.copy()
    if drop_x_label and drop_x_label in xticks:
        xticks[xticks.index(drop_x_label)] = ""
    if drop_y_label and drop_y_label in yticks:
        yticks[yticks.index(drop_y_label)] = ""

    # Plot
    mask = np.triu(np.ones_like(rho, dtype=bool))
    plt.figure(figsize=(11, 9))
    ax = sns.heatmap(
        rho,
        mask=mask,
        annot=False,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        xticklabels=xticks,
        yticklabels=yticks,
        cbar_kws={"label": None}
    )

    # Add bold correlation values and stars
    for i in range(len(spike_features)):
        for j in range(len(spike_features)):
            if i > j:
                r_val = rho.iloc[i, j]
                p = pval.iloc[i, j]
                sig = p < 0.05
                fontweight = 'bold' if sig else 'normal'
                ax.text(j + 0.5, i + 0.5, f"{r_val:.2f}",
                        ha='center', va='center',
                        fontsize=13, fontweight=fontweight, color='black')
                if stars.iloc[i, j] != "":
                    ax.text(j + 0.5, i + 0.3, stars.iloc[i, j],
                            ha='center', va='bottom', color='black',
                            fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=13)
    plt.yticks(rotation=0, fontsize=13)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


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


