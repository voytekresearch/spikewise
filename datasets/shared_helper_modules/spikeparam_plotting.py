import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from scipy.stats import pearsonr


def plot_corr_heatmap(
    df_features,
    calculate_corr=True,
    type_heatmap="half",
    show_sig=True,
    cmap="coolwarm"
):
    """
    Plots correlation heatmap of spike and LFP features with optional significance stars.

    Args:
        df_features: pandas DataFrame (features OR precomputed correlation matrix).
        calculate_corr: If True, compute Pearson correlations & p-values.
        type_heatmap: "half" or "full" view.
        show_sig: Whether to overlay significance stars.
        cmap: Color map for heatmap.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr

    print("📊 Starting heatmap generation...")

    if calculate_corr:
        print("✅ Calculating correlation and p-values...")
        rho = df_features.corr()
        pval = df_features.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(len(df_features.columns))
    else:
        print("🧮 Using precomputed correlation matrix.")
        rho = df_features
        pval = None

    if show_sig and pval is not None:
        print("✨ Adding significance stars to annotations...")
        def p_to_star(p):
            if p < 0.001: return '***'
            elif p < 0.01: return '**'
            elif p < 0.05: return '*'
            else: return ''
        stars = pval.applymap(p_to_star)
        annot = (rho.round(2).astype(str) + stars).values  # Force to numpy array
    else:
        print("📉 Skipping significance stars.")
        annot = rho.round(2).values

    print(f"🧾 Annotation matrix shape: {annot.shape} | dtype: {type(annot)}")

    # Check if half-matrix is possible
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

