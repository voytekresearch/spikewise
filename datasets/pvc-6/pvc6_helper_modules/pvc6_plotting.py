import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

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