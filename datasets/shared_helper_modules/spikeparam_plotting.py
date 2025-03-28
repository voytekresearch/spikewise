import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from scipy.stats import pearsonr


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def plot_corr_heatmap(
    df_features,
    spike_features,
    lfp_features,
    calculate_corr=False,
    type_heatmap="half",
    show_sig=True,
    cmap="coolwarm"
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
    """

    print("📊 Starting heatmap generation...")

    # Select only relevant features
    df_selected = df_features[spike_features + lfp_features]

    # ✅ Clean data: Remove rows with NaN or Inf
    print("🧼 Cleaning data by dropping NaN and Inf values...")
    df_cleaned = df_selected.replace([np.inf, -np.inf], np.nan).dropna()

    if df_cleaned.empty:
        raise ValueError("❗ After cleaning, no data remains. Check for excessive missing values.")

    print(f"✅ Data cleaned: {len(df_cleaned)} rows remaining out of {len(df_features)}.")

    if calculate_corr:
        print("✅ Calculating correlation and p-values...")
        rho = df_cleaned.corr()
    else:
        print("🧮 Using precomputed correlation matrix.")
        rho = df_features.loc[spike_features, lfp_features]

    # Compute p-values using Pearson correlation
    print("🧪 Calculating p-values for significance stars...")
    pval = pd.DataFrame(
        np.zeros_like(rho),
        index=rho.index,
        columns=rho.columns
    )
    
    for row in rho.index:
        for col in rho.columns:
            if row != col:
                r, p = pearsonr(df_cleaned[row], df_cleaned[col])
                pval.at[row, col] = p
    
    # Convert p-values to stars
    if show_sig:
        print("✨ Adding significance stars to annotations...")
        def p_to_star(p):
            if p < 0.001: return '***'
            elif p < 0.01: return '**'
            elif p < 0.05: return '*'
            else: return ''
        
        stars = pval.applymap(p_to_star)
        annot = (rho.round(2).astype(str) + stars).values
    else:
        print("📉 Skipping significance stars.")
        annot = rho.round(2).values

    print(f"🧾 Annotation matrix shape: {annot.shape} | dtype: {type(annot)}")

    # Handle half matrix only if square
    if type_heatmap == "half" and rho.shape[0] == rho.shape[1]:
        print("📐 Applying upper triangle mask (half heatmap).")
        mask = np.triu(np.ones_like(rho, dtype=bool))
        square = True
    else:
        if type_heatmap == "half":
            print("⚠️ Matrix not square — switching to full heatmap.")
        mask = None
        square = False

    print("🎨 Drawing heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        rho,
        mask=mask,
        annot=annot,
        fmt="",
        cmap=cmap,
        linewidths=0.5,
        center=0,
        square=square,
        annot_kws={"size": 12}
    )
    plt.xticks(fontsize=14, rotation=45, ha='right')
    plt.yticks(fontsize=14)
    plt.title("Correlation Heatmap with Significance")
    plt.tight_layout()
    plt.show()

    print("✅ Done!")
