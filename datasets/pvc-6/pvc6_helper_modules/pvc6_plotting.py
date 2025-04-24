import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import warnings

warnings.filterwarnings('ignore')

def plot_pink_spikes(sp, indices_to_plot):
    """
    Plots pink spikes with specified indices.

    Args:
        sp: Spike data.
        indices_to_plot: Indices of spikes to plot.
    """
    sp.plot(indices_to_plot, color='hotpink', mode='full', show_points=True)
    fig = plt.gcf()
    ax = plt.gca()
    
    ax.set_xlabel('Time (ms)', fontsize=18)
    ax.set_ylabel('Voltage', fontsize=18)
    
    legend = ax.get_legend()
    if legend:
        plt.setp(legend.get_texts(), fontsize=18)
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    for line in ax.get_lines():
        line.set_linewidth(3)
    
    plt.show()

def plot_correlation_scatter(df_pink_filtered):
    """
    Plots correlation scatter plots for specified features.

    Args:
        df_pink_filtered: DataFrame containing the data.
    """
    fig, axs = plt.subplots(1, 3, figsize=(19, 6))
    
    axs[0].plot(df_pink_filtered['exp_const'], df_pink_filtered['stim_mean'], '.C6', markersize=20)
    axs[0].set_xlabel('Decay exp constant', fontsize=30)
    axs[0].set_ylabel('Stim mean', fontsize=30)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].tick_params(axis='both', labelsize=20)
    
    axs[1].plot(df_pink_filtered['peak_sharpness'], df_pink_filtered['stim_mean'], '.C5', markersize=20)
    axs[1].set_xlabel('Peak sharpness', fontsize=30)
    axs[1].set_ylabel('Stim mean', fontsize=30)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].tick_params(axis='both', labelsize=20)
    
    axs[2].plot(df_pink_filtered['stim_mean'], df_pink_filtered['log_isi'], '.C7', markersize=20)
    axs[2].set_xlabel('Log isi', fontsize=30)
    axs[2].set_ylabel('Stim mean', fontsize=30)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].tick_params(axis='both', labelsize=20)
    
    plt.tight_layout()
    plt.show()

def plot_avg_waveform_by_stim_type(all_constant_spks, all_ramp_spks, all_pink_spks):
    """
    Plots average waveforms for different stimulus types.

    Args:
        all_constant_spks: Constant spike data.
        all_ramp_spks: Ramp spike data.
        all_pink_spks: Pink spike data.
    """
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
    plt.title('Mean and Standard Deviation of Spikes')
    plt.xlim(1600, 2400)
    plt.show()



def plot_confusion_matrix(best_model, X_test, y_test):
    """
    Plots confusion matrix for a model.

    Args:
        best_model: Trained model.
        X_test: Test feature set.
        y_test: Test labels.
    """
    conf_matrix = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    conf_matrix.plot(cmap='Blues', ax=ax, xticks_rotation=45, values_format='d')
    
    for text in ax.texts:
        text.set_fontsize(14)
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=17)
    plt.ylabel('True Label', fontsize=17)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

def plot_ridge_results(y, ridge_results, title="Ridge Regression Results"):
    """
    Plots results of ridge regression.

    Args:
        y: Actual values.
        ridge_results: Dictionary containing ridge regression results.
        title: Plot title.
    """
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

    color_mapping = {
        'peak_amp': 'C5',
        'exp_const': 'C6',
        'exp_lambda': 'C6',
        'inflection_amp': 'C3',
        'ramp_amp': 'C4',
        'inflection_time': 'C3',
        'peak_sharpness': 'C5',
        'peak_width': 'C5',
        'log_isi': 'C7'
    }
    
    feature_importance_df['Color'] = feature_importance_df['Feature'].apply(
        lambda x: 'hotpink' if x.startswith('stim_') else color_mapping.get(x, 'gray')
    )

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred_cv, s=55)
    plt.xlabel("Actual Values", fontsize=20)
    plt.ylabel("Predicted Values", fontsize=20)
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    m, b = np.polyfit(y, y_pred_cv, 1)
    plt.plot(y, m * y + b, color='red', linewidth=2)  
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Coefficient', y='Feature', data=feature_importance_df, palette=feature_importance_df['Color']
    )

    plt.title('Feature Importance (Ridge Regression)', fontsize=20)
    plt.xlabel('Coefficient', fontsize=20)
    plt.ylabel('Feature', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

    significant_features = feature_importance_df[feature_importance_df["p-value"] < 0.05]
    
    if not significant_features.empty:
        print("\nSignificant Features (p < 0.05):")
        print(significant_features[["Feature", "Coefficient", "CI Lower", "CI Upper", "p-value"]])
    else:
        print("\nNo features were statistically significant (p < 0.05).")

    bootstrapped_coefs = ridge_results["bootstrapped_coefs"]
    
    plt.figure(figsize=(12, 6))
    
    coef_df = pd.DataFrame(bootstrapped_coefs, columns=feature_names)
    coef_df = coef_df.melt(var_name='Feature', value_name='Coefficient Value')
    
    coef_df['Color'] = coef_df['Feature'].apply(
        lambda x: 'hotpink' if x.startswith('stim_') else color_mapping.get(x, 'gray')
    )
    
    sns.violinplot(
        x='Feature', 
        y='Coefficient Value', 
        data=coef_df, 
        palette=coef_df['Color'].unique(), 
        inner="point", 
        scale="width"
    )
    
    plt.xticks(rotation=45)
    plt.title("Bootstrapped Coefficient Distributions", fontsize=20)
    plt.ylabel("Coefficient Value", fontsize=16)
    plt.xlabel("Feature", fontsize=16)
    plt.show()

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.boxplot(x=r2_scores)
    plt.title("K-Fold R² Scores", fontsize=16)
    plt.xlabel("R²", fontsize=14)

    plt.subplot(1, 3, 2)
    sns.histplot(bootstrapped_r2, kde=True, bins=30)
    plt.axvline(r2_mean, color='red', linestyle='--', label=f"Mean R²: {r2_mean:.3f}")
    plt.axvline(r2_ci[0], color='gray', linestyle=':', label=f"95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")
    plt.axvline(r2_ci[1], color='gray', linestyle=':')
    plt.title("Bootstrapped R² Distribution", fontsize=16)
    plt.xlabel("R²", fontsize=14)
    plt.legend()

    plt.subplot(1, 3, 3)
    sns.histplot(bootstrapped_adjusted_r2, kde=True, bins=30)
    plt.axvline(adjusted_r2_mean, color='red', linestyle='--', label=f"Mean Adjusted R²: {adjusted_r2_mean:.3f}")
    plt.axvline(adjusted_r2_ci[0], color='gray', linestyle=':', label=f"95% CI: [{adjusted_r2_ci[0]:.3f}, {adjusted_r2_ci[1]:.3f}]")
    plt.axvline(adjusted_r2_ci[1], color='gray', linestyle=':')
    plt.title("Bootstrapped Adjusted R² Distribution", fontsize=16)
    plt.xlabel("Adjusted R²", fontsize=14)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"\nMean Adjusted R-squared (Bootstrapped): {adjusted_r2_mean:.3f}")
    print(f"95% CI for Adjusted R-squared: {adjusted_r2_ci}")



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
    plt.title('Combined Scatter Plot of Actual vs. Predicted Values for All Models', fontsize=20)
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
    plt.title('Combined Feature Importance for All Models', fontsize=20)
    plt.xlabel('Coefficient (Importance)', fontsize=20)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=20)  # Increase font size for the feature names
    
    # Disable the legend
    plt.legend([], [], frameon=False)
    
    # Show the plot
    plt.show()


# Function to plot feature importance
def plot_feature_importance_categorical(best_model, X):
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        raise AttributeError("The model does not have feature_importances_ attribute.")
    
    feature_names = X.columns
    df_importance = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_importance = df_importance.sort_values(by="Importance", ascending=False)
    
    color_mapping = {
        'peak_amp': 'C5',
        'exp_const': 'C6',
        'exp_lambda': 'C6',
        'inflection_amp': 'C3',
        'ramp_amp': 'C4',
        'inflection_time': 'C3',
        'peak_sharpness': 'C5',
        'peak_width': 'C5',
        'log_isi': 'C7'
    }
    df_importance['Color'] = df_importance['Feature'].map(color_mapping).fillna('#8c8c8c')

    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=df_importance, palette=df_importance['Color'].tolist())
    plt.title("Feature Importance (Random Forest)", fontsize=20)
    plt.xlabel("Importance", fontsize=16)
    plt.ylabel("Feature", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()

# Function to plot bootstrapped accuracies as histograms
def plot_bootstrap_histograms(bootstrapped_results, model_names):
    plt.figure(figsize=(18, 6))
    for i, (accs, model_name) in enumerate(zip(bootstrapped_results, model_names)):

        # Compute statistics for R² and Adjusted R²
        accs_mean = np.mean(accs)
        accs_ci = np.percentile(accs, [2.5, 97.5])
        
        plt.subplot(1, 3, i+1)
        sns.histplot(accs, kde=True, bins=30)
        plt.axvline(accs_mean , color='red', linestyle='--', label=f"Mean R²: {accs_mean:.3f}")
        plt.axvline(accs_ci[0], color='gray', linestyle=':', label=f"95% CI: [{accs_ci[0]:.3f}, {accs_ci[1]:.3f}]")
        plt.axvline(accs_ci[1], color='gray', linestyle=':')
        plt.title(f"{model_name} Bootrstrapped Accuracy Distribution", fontsize=16)
        plt.xlabel("Accuracy", fontsize=14)
        plt.ylabel("Frequency", fontsize=16)
        plt.legend()
    plt.tight_layout()
    plt.show()



