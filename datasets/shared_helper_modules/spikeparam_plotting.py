import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from scipy.stats import pearsonr

def plot_corr_heatmap(df_features, calculate_corr = True,type_heatmap="half"):
    """
    Plots correlation heatmap of spike paramaterization features.

    Args:
        df_features: pandas dataframe with features 
        calculate_corr: calculates correlations between features when True (defualt). If False, user should input correlations. 
        type_heatmap: plot full matrix "full" or half matrix "half" (default)
       
    """

    if calculate_corr:
        # Calculate correlation matrix and p-values
        rho = df_features.corr()
        pval = df_features.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
        p = pval < 0.05  # Create a mask for significant p-values

    else:
        rho = df_features

    if type_heatmap == "half":
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(rho, dtype=bool))
        
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(rho, mask=mask, cmap='coolwarm', annot=True, fmt='.2', linewidths=0.5, center=0, square=True,  annot_kws={"size": 12})
        # Increase fontsize of tick labels on both axes
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.title('Correlation Heatmap')
       
        
    else:
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(rho, cmap='coolwarm', annot=True, fmt='.2', linewidths=0.5, center=0, square=True,  annot_kws={"size": 12})
        # Increase fontsize of tick labels on both axes
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.title('Correlation Heatmap')
  



    plt.show()
    