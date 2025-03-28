import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from scipy.stats import pearsonr


def plot_corr_heatmap(
    df_features,
    spike_features,
    lfp_features,
    calculate_corr=False,
    type_heatmap="half",
    show_sig=True,
    cmap="coolwarm",
    corr_threshold=0.1  # Correlation threshold for plotting stars
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
    """

    # Select only relevant features
    df_selected = df_features[spike_features + lfp_features]

    # Clean data by removing NaN and Inf values
    df_cleaned = df_selected.replace([np.inf, -np.inf], np.nan).dropna()
    if df_cleaned.empty:
        raise ValueError("After cleaning, no data remains. Check for excessive missing values.")

    # Calculate correlation matrix and extract only spike vs LFP correlations
    if calculate_corr:
        correlation_matrix = df_cleaned.corr()
    else:
        correlation_matrix = df_features.corr()

    # ✅ Extract the correct matrix with spikes on Y and LFPs on X
    rho = correlation_matrix.loc[spike_features, lfp_features]

    # Compute p-values using Pearson correlation
    pval = pd.DataFrame(np.zeros_like(rho), index=spike_features, columns=lfp_features)
    for row in spike_features:
        for col in lfp_features:
            r, p = pearsonr(df_cleaned[row], df_cleaned[col])
            pval.at[row, col] = p

    # Convert p-values to stars (show only if correlation > threshold)
    if show_sig:
        def p_to_star(p, r):
            if abs(r) < corr_threshold:
                return ""  # No stars for low correlations
            if p < 0.001: return '***'
            elif p < 0.01: return '**'
            elif p < 0.05: return '*'
            else: return ''
        stars = pval.copy()
        for i in range(len(spike_features)):
            for j in range(len(lfp_features)):
                stars.iloc[i, j] = p_to_star(pval.iloc[i, j], rho.iloc[i, j])

    # Plotting adjustments
    plt.figure(figsize=(14, 12))

    # Plot the heatmap
    sns.heatmap(
        rho,
        annot=rho.round(2),
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        center=0,
        xticklabels=lfp_features,
        yticklabels=spike_features,
        annot_kws={"size": 8, "color": "black"}
    )

    # ✅ Plot significance stars on top
    if show_sig:
        for i in range(len(spike_features)):
            for j in range(len(lfp_features)):
                if stars.iloc[i, j] != "":
                    plt.text(j + 0.5, i + 0.5, stars.iloc[i, j],
                             ha='center', va='center', color='black', fontsize=10)

    # Adjust axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.title("Correlation Heatmap with Significance")
    plt.tight_layout()
    plt.show()

