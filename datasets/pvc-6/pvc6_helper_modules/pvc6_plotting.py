import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

def plot_pink_spikes(sp, indices_to_plot):
    sp.plot(indices_to_plot, color='hotpink', mode = 'full', show_points=True)

    # After plotting, get the current figure and axis
    fig = plt.gcf()  # Get the current figure
    ax = plt.gca()   # Get the current axis
    
    # Modify the font size of the labels
    ax.set_xlabel('Time (ms)', fontsize=18)
    ax.set_ylabel('Voltage', fontsize=18)
    
    # Modify the font size of the legend
    legend = ax.get_legend()
    if legend:
        plt.setp(legend.get_texts(), fontsize=18)
    # Modify the font size of the tick numbers
    ax.tick_params(axis='both', which='major', labelsize=16)  # Adjust tick label size for both axes
    
    
    # Modify line thickness
    for line in ax.get_lines():
        line.set_linewidth(3)
    
    # Finally, re-display the updated plot
    plt.show()


def plot_correlation_scatter(df_pink_filtered):
    # Create a figure with 3 subplots side by side
    fig, axs = plt.subplots(1, 3, figsize=(19, 6))  # Adjust the size as needed
    
    # First plot: exp_lambda vs stim_mean
    axs[0].plot(df_pink_filtered['exp_const'], df_pink_filtered['stim_mean'], '.C6', markersize=20)  # Blue dots
    
    axs[0].set_xlabel('Decay exp constant', fontsize=30)
    axs[0].set_ylabel('Stim mean', fontsize=30)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    axs[0].tick_params(axis='both', labelsize=20)
    
    
    # First plot: exp_lambda vs stim_mean
    axs[1].plot(df_pink_filtered['peak_sharpness'], df_pink_filtered['stim_mean'], '.C5', markersize=20)  # Blue dots
    
    axs[1].set_xlabel('Peak sharpness', fontsize=30)
    axs[1].set_ylabel('Stim mean', fontsize=30)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].tick_params(axis='both', labelsize=20)
                 
    # Third plot: stim_mean vs log_isi
    axs[2].plot(df_pink_filtered['stim_mean'], df_pink_filtered['log_isi'], '.C7', markersize=20) 
    
    axs[2].set_xlabel('Log isi', fontsize=30)
    axs[2].set_ylabel('Stim mean', fontsize=30)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].tick_params(axis='both', labelsize=20)

    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()



def plot_avg_waveform_by_stim_type(all_contant_spks, all_ramp_spks,all_pink_spks):
    # Calculate mean and standard deviation for each array
    mean_constant_spks = np.mean(all_contant_spks, axis=0)
    std_constant_spks = np.std(all_contant_spks, axis=0)
    
    mean_ramp_spks = np.mean(all_ramp_spks, axis=0)
    std_ramp_spks = np.std(all_ramp_spks, axis=0)
    
    mean_pink_spks = np.mean(all_pink_spks, axis=0)
    std_pink_spks = np.std(all_pink_spks, axis=0)
    
    # Plot the mean and standard deviation
    plt.figure(figsize=(10, 6))
    
    # Plot for all_constant_spks
    plt.plot(mean_constant_spks, label='Mean Constant Spikes', color='green', linewidth=4)
    plt.fill_between(range(len(mean_constant_spks)), mean_constant_spks - std_constant_spks, mean_constant_spks + std_constant_spks, color='green', alpha=0.3)
    
    # Plot for all_ramp_spks
    plt.plot(mean_ramp_spks, label='Mean Ramp Spikes', color='purple', linewidth=4)
    plt.fill_between(range(len(mean_ramp_spks)), mean_ramp_spks - std_ramp_spks, mean_ramp_spks + std_ramp_spks, color='purple', alpha=0.3)
    
    # Plot for all_pink_spks
    plt.plot(mean_pink_spks, label='Mean Pink Spikes', color='hotpink', linewidth=4)
    plt.fill_between(range(len(mean_pink_spks)), mean_pink_spks - std_pink_spks, mean_pink_spks + std_pink_spks, color='lightpink', alpha=0.3)
    
    # Set plot labels and title
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage')
    plt.title('Mean and Standard Deviation of Spikes')
    #plt.legend()
    plt.xlim(1600,2400)



def plot_feature_importance_categorical(best_model,X, X_train):
    # Define preprocessing steps
    numeric_features = X.select_dtypes(include=['float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    # Visualize feature importance
    importances = best_model.named_steps['classifier'].feature_importances_
    
    # Fit OneHotEncoder on the categorical features using training data
    best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].fit(X_train[categorical_features])
    
    # Get feature names for categorical features after one-hot encoding
    feature_names_cat = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    
    # Combine encoded column names with numeric feature names
    feature_names = list(feature_names_cat) + list(numeric_features)
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    
    # Define color mapping
    color_mapping = {
        'peak_amp': 'C5', # Example colors
        'exp_const': 'C6',
        'exp_lambda': 'C6',
        'inflection_amp': 'C3',
        'ramp_amp': 'C4',
        'inflection_time': 'C3',
        'peak_sharpness': 'C5',
        'peak_width': 'C5',
        'log_isi': 'C7'
        
}
    
    # Apply color mapping
    feature_importance_df['Color'] = feature_importance_df['Feature'].map(color_mapping)
    feature_importance_df['Color'].fillna('gray', inplace=True)  # Assign default color to missing values
    
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette=feature_importance_df['Color'])
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance', fontsize=20)
    plt.ylabel('Feature', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


def plot_confusion_matrix(best_model, X_test, y_test):

    # Generate confusion matrix
    conf_matrix = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    conf_matrix.plot(cmap='Blues', ax=ax, xticks_rotation=45, values_format='d')  # Plot the confusion matrix
    
    # Adjust font size for annotations (numbers within the matrix)
    for text in ax.texts:
        text.set_fontsize(14)  # Set the font size for annotations
    
    
    
    plt.title('Confusion Matrix', fontsize=16)  # Increase the font size for title
    plt.xlabel('Predicted Label', fontsize=17)  # Increase the font size for x-axis label
    plt.ylabel('True Label', fontsize=17)       # Increase the font size for y-axis label
    plt.xticks(fontsize=16)  # Increase the font size for x-axis ticks
    plt.yticks(fontsize=16)  # Increase the font size for y-axis ticks
    
    plt.show()





def plot_ridge_results(y, ridge_results, title="Ridge Regression Results"):
    """
    Plots Actual vs Predicted values, Feature Importance, Bootstrapped Coefficients with CIs,
    and Bootstrapped R² distribution.

    Args:
        y (pd.Series): Actual values.
        ridge_results (dict): Dictionary containing model results from `run_ridge_regression_kfold()`.
        title (str): Title for the scatter plot.
    """

    y_pred_cv = ridge_results["y_pred_cv"]
    coefficients = ridge_results["coefficients"]
    feature_names = ridge_results["feature_names"]
    ci_lower = ridge_results["ci_lower"]
    ci_upper = ridge_results["ci_upper"]
    p_values = ridge_results["p_values"]
    r2_scores = ridge_results["r2_scores"]
    bootstrapped_r2 = ridge_results["bootstrapped_r2"]
    r2_mean = ridge_results["r2_mean"]
    r2_ci = ridge_results["r2_ci"]

    # Create DataFrame for plotting
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names, 
        'Coefficient': coefficients,
        'CI Lower': ci_lower,
        'CI Upper': ci_upper,
        'p-value': p_values
    })
    
    feature_importance_df = feature_importance_df.sort_values(by='Coefficient', ascending=False)

    # 1. PLOT ACTUAL vs PREDICTED
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred_cv, s=55)
    plt.xlabel("Actual Values", fontsize=20)
    plt.ylabel("Predicted Values", fontsize=20)
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Regression Line
    m, b = np.polyfit(y, y_pred_cv, 1)
    plt.plot(y, m * y + b, color='red', linewidth=2)  
    plt.show()

    # 2. PLOT FEATURE IMPORTANCE with CIs
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Coefficient', y='Feature', data=feature_importance_df
    )

    plt.title('Feature Importance (Ridge Regression)', fontsize=20)
    plt.xlabel('Coefficient', fontsize=20)
    plt.ylabel('Feature', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

    # 3. PRINT SIGNIFICANT FEATURES
    significant_features = feature_importance_df[feature_importance_df["p-value"] < 0.05]
    
    if not significant_features.empty:
        print("\nSignificant Features (p < 0.05):")
        print(significant_features[["Feature", "Coefficient", "CI Lower", "CI Upper", "p-value"]])
    else:
        print("\nNo features were statistically significant (p < 0.05).")

    # 4. PLOT BOOTSTRAPPED DISTRIBUTIONS
    bootstrapped_coefs = ridge_results["bootstrapped_coefs"]

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=bootstrapped_coefs, inner="point", scale="width")
    plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=45)
    plt.title("Bootstrapped Coefficient Distributions", fontsize=20)
    plt.ylabel("Coefficient Value", fontsize=16)
    plt.show()

    # 5. PLOT K-FOLD AND BOOTSTRAPPED R²
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=r2_scores)
    plt.title("K-Fold R² Scores", fontsize=16)
    plt.xlabel("R²", fontsize=14)

    plt.subplot(1, 2, 2)
    sns.histplot(bootstrapped_r2, kde=True, bins=30)
    plt.axvline(r2_mean, color='red', linestyle='--', label=f"Mean R²: {r2_mean:.3f}")
    plt.axvline(r2_ci[0], color='gray', linestyle=':', label=f"95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")
    plt.title("Bootstrapped R² Distribution", fontsize=16)
    plt.xlabel("R²", fontsize=14)
    plt.legend()

    plt.tight_layout()
    plt.show()

