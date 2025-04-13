'''
Author: David Brin
Date Created: 4/2/2025

This script can be run to group and analyze the given DataFrame, generating relevant plots and statistical summaries.  
It includes three main functions:  
- A function for grouping the data based on specified parameters.  
- A function for generating combined plots to visualize trends across groups.  
- A function for computing statistical comparisons between groups to identify significant results.  

'''
import pandas as pd
import numpy as np
from itertools import combinations
from IPython.display import display
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math
#  taken from 'stats_from_allMonkey_filt'  --> unused imports (might be some unused imports but it definetly covers the bases)
import sys
sys.path.append(r'..\..\..\spikeparam')    #<-- to access the spikeparam functions from inside the repository
from spikeparam.patch.fit import Spike
from spikeparam.patch.fit import SpikeGroup
from neurodsp import spectral
from scipy import signal
import scipy
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
from neurodsp import filt
from neurodsp.timefrequency import amp_by_time, phase_by_time
from neurodsp.plts import plot_time_series, plot_instantaneous_measure
from neurodsp.plts.time_series import plot_bursts
from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats
from scipy.signal import sosfiltfilt, butter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy.spatial as sp_spatial
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os
from fooof import FOOOF
sns.set(rc={'figure.figsize':(12,9)})
sns.set_style('whitegrid')
sns.set_style("whitegrid", {'axes.grid' : False})
import IProgress
import openpyxl
import glob
import pickle
plt.rcParams["figure.dpi"] = 200 


# Functions to save a datrame to a pickle file and another to extract the data from the pickle file

def save_dataframe_to_pickle(dataframe, file_path):
    """
    Function to save a DataFrame as a pickle file.
    
    Args:
    - dataframe (pd.DataFrame): DataFrame to be saved.
    - file_path (str): Path to save the pickle file.
    """
    dataframe.to_pickle(file_path)
    print(f"Data frame saved to {file_path}")

def load_dataframe_from_pickle(file_path):
    """
    Function to extract a DataFrame from a pickle file.
    
    Args:
    - file_path (str): Path to the pickle file.
    
    Returns:
    - dataframe (pd.DataFrame): Loaded DataFrame.
    """
    dataframe = pd.read_pickle(file_path)
    return dataframe

def save_dict_to_pickle(dictionary, filepath):
    """
    Save a dictionary to a pickle file.

    Parameters:
        dictionary (dict): The dictionary to save.
        filepath (str): The path to the pickle file.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(dictionary, f)
    print(f"Dictionary saved to {filepath}")


def load_dict_from_pickle(filepath):
    """
    Load a dictionary from a pickle file.

    Parameters:
        filepath (str): The path to the pickle file.

    Returns:
        dict: The loaded dictionary.
    """
    with open(filepath, 'rb') as f:
        dictionary = pickle.load(f)
    print(f"Dictionary loaded from {filepath}")
    return dictionary



#GROUPING FUNCTION BELOW


# Identify metadata columns
metadata_cols = ['dendriticType', 'SomaLayerLoc', 'brainOrigin', 'Sex', 'Species']

# Function to create subcomparisons for varying metadata
def create_comparison_groups(df):
    '''
    Creates the comparisons and RETURNS a dictionary of all the sub dataframes created from the groups (and the grouping data frame)

    df- full df to be split into groups
    metadata_cols - all unique metadata types to be grouped by (no longer a param but set right above function)
    Returns: grouping_results_df (dataframe of grouping info), sub_dfs_dict (dictionary of each split group data frame)
    '''
    
    results = []

    for varying_col in metadata_cols:
        # Columns to keep fixed (all except the varying one)
        fixed_cols = [col for col in metadata_cols if col != varying_col]

        # Group data by fixed metadata columns
        grouped = df.groupby(fixed_cols)
        subcomparison_index = 1

        for group_name, group_data in grouped:
            # Get unique values for the varying column within the fixed group
            varying_groups = group_data[varying_col].unique()

            # Ensure there are at least two values for the varying column
            if len(varying_groups) > 1:
                # Create a single subcomparison that includes all varying groups
                subsets = {value: group_data[group_data[varying_col] == value] for value in varying_groups}
                
                results.append({
                    'Subcomparison': subcomparison_index,
                    'Varying Metadata': varying_col,
                    'Fixed Metadata': ', '.join([f"{col}={val}" for col, val in zip(fixed_cols, group_name)]) if isinstance(group_name, tuple) else f"{fixed_cols[0]}={group_name}",
                    'Groups': {value: len(subset) for value, subset in subsets.items()}
                })

                subcomparison_index += 1
    grouping_results_df = pd.DataFrame(results)

    
    sub_dfs_dict = {}  # Dictionary to store sub dataframes
    rejected = 0
    for _, row in grouping_results_df.iterrows():
        # Extract fixed metadata as a dictionary
        fixed_metadata = {}
        for item in row['Fixed Metadata'].split(", "):
            key, val = item.split("=")
            val = val if "." not in val or not val.replace(".", "", 1).isdigit() else float(val)
            fixed_metadata[key.strip()] = val
            
        varying_metadata_str = row['Varying Metadata']  
        dict_key = f"{varying_metadata_str} {row['Subcomparison']}"
    
        # Filter df for rows matching the fixed metadata
        condition = pd.Series(True, index=df.index)
        for col, val in fixed_metadata.items():
            condition &= df[col] == val
    
        sub_df = df[condition]
        if sub_df.shape[0] == 0:  
            print(f" No match for fixed_metadata!!")
            rejected += 1
            print(f"number rejected: {rejected}")
            continue
    
        # Store sub_df in the dictionary
        sub_dfs_dict[dict_key] = sub_df
    
    display(len(sub_dfs_dict))
    return grouping_results_df, sub_dfs_dict
    

'''
example usage:
grouping_results_df, sub_dfs_dict = create_comparison_groups(allMonkey_df)

if len(grouping_results_df) > 50:                                      <-- for easier readability at lengths larger than 50
    for i in range(0, len(grouping_results_df), 50):
        display(grouping_results_df.iloc[i:i+50])
else:
    display(grouping_results_df)

'''


# STATS FUNCTION BELOW   (one is there to show how stats will be printed and one to print ALL stats)

def print_stats(df, condition, anova = True, postH = True):
    spike_features = ['ramp_amp', 'inflection_time', 'inflection_amp', 'peak_amp', 'peak_width', 
                               'peak_sharpness', 'exp_lambda', 'exp_const', 'log_isi', 'r_squared_ramp']
    '''
    prints ANOVA and/or PostHoc Tukey results for each feature
    params: df- the data frame to look through (for groups)
            condition- the non-fixed metadata variable
    returns: nothing, prints results
    '''
    if anova:
        print("=== ANOVA Results ===")
        anova_results = {feature: {} for feature in spike_features}
        
        for feature in spike_features:
            # Prepare groups: one list of Series per condition
            if feature == 'log_isi':
                groups = [
                    df[df[condition] == cond][feature].dropna()
                    for cond in df[condition].unique()
                ]
            else:
                groups = [
                    df[df[condition] == cond][feature]
                    for cond in df[condition].unique()
                ]
            
            # Skip the ANOVA if any group is empty (e.g., all NaNs)
            if any(len(group) == 0 for group in groups):
                print(f"  Skipping ANOVA for {feature} due to empty group(s).")
                continue

            try:
                anova_results[feature][condition] = scipy.stats.f_oneway(*groups)
            except Exception as e:
                print(f"  Error running ANOVA for {feature}: {e}")
        
        # Print results
        for feature, results in anova_results.items():
            if condition in results:
                f_value, p_value = results[condition]
                print(f"  {feature:20s} | F = {f_value:.4f}, p = {p_value:.4e}")
    if postH:
        print("===PostHoc Tukey Results===")
        # Initialize a dictionary to store Tukey's test results
        tukey_results = {feature: None for feature in spike_features}
        # Perform ANOVA and Tukey's test for each spike feature against the given condition
        for feature in spike_features:
            if df.empty or df[condition].dropna().empty:
                print(f"Skipping due to empty data for condition: {condition}")
                continue
            if df[feature].dropna().shape[0] < 2 or df[feature].dropna().nunique() < 2:
                print(f"Skipping {feature}: not enough data or no variation.")
                continue

            model_formula = f'{feature} ~ C({condition})'
            

            model = smf.ols(model_formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            # If ANOVA shows significant results, perform Tukey's test
            if anova_table["PR(>F)"].iloc[0] < 0.05:
                try:
                    tukey = pairwise_tukeyhsd(endog=df[feature].astype(float),
                                              groups=df[condition].astype(str),
                                              alpha=0.05)
                    tukey_results[feature] = tukey.summary()
                except Exception as e:
                    print(f'Error performing Tukey\'s test for {feature} under {condition}: {e}')
    
        # Print the Tukey's test results for each feature
        for feature, result in tukey_results.items():
            print(f'Tukey\'s test results for {feature}:')
            if result is not None:
                print(result)
            else:
                print(f'No significant differences found for {feature}.')
            print('\n')


def printAll_stats(sub_dfs_dict, anova = True, postH = True):
    '''
    Prints all stats shown above
    Pass through the dictionary with sub dataframes
    Returns nothing
    '''
    if not(anova or postH):
        print("What are we even doing here then?")           #usage
        return
    metadata_cols = ["dendriticType", "SomaLayerLoc", "brainOrigin", "Species", "Sex" ]
    index = 0
    for key in sub_dfs_dict:
        sub_df = sub_dfs_dict[key]                #while iterating through dictionary, get varying metadata
        print(key)
        index +=1
        varying_metadata = next(
            (col for col in metadata_cols if sub_df[col].nunique() > 1), None 
        )
        
        if varying_metadata is None:
            print("No varying metadata found for this group.") 
            continue
        print(f'CONDITION:  {varying_metadata}\n')
        if anova and postH:
            print_stats(sub_df, varying_metadata, anova = True, postH = True)
        elif anova:
            print_stats(sub_df, varying_metadata, anova = True, postH = False)
        else:
            print_stats(sub_df, varying_metadata, anova = False, postH = True)



#          PLOTS    (combined)

                #split into 2 base functions and 1 calling function

def plot_waveforms_grid(sub_dfs_dict, param, combined_dict_filt, median=False):
    """
    Plots overlapping waveforms for all sub_dfs filtered by a specific parameter, ensuring that 
    waveforms for different types are plotted together in a single figure per DataFrame.
    The y-axis is set to a unified scale based on the global min/max values across all spikes.

    Parameters:
    - sub_dfs_dict (dict): Dictionary where keys are sub_df names and values are DataFrames.
    - param (str): The metadata parameter to vary across traces.
    - combined_dict_filt (dict): Dictionary where keys are formatted as "{file_name} {Sweep_#}" 
      and values are arrays of spike data.
    - median (bool, optional): If True, calculates the median spike waveform instead of the mean.

    Returns:
    None: The function displays the plots.
    """
    filtered_dfs = {key: df for key, df in sub_dfs_dict.items() if key.startswith(param)}
    if not filtered_dfs:
        print(f"No data frames found for param: {param}")
        return

    # Collect all unique values of param across all DataFrames
    all_unique_values = set()
    for sub_df in filtered_dfs.values():
        all_unique_values.update(sub_df[param].unique())

    # Generate a consistent color mapping
    cmap = plt.get_cmap('Dark2')  
    norm = mcolors.Normalize(vmin=0, vmax=len(all_unique_values) - 1)
    color_map = {value: cmap(norm(i)) for i, value in enumerate(all_unique_values)}

    # Collect all spike min/max values for global scaling
    all_min_values = []
    all_max_values = []
    for sub_df in filtered_dfs.values():
        for _, row in sub_df.iterrows():
            spike_key = row['Spike_ID']
            if spike_key in combined_dict_filt:
                spike_waveform = np.array(combined_dict_filt[spike_key])
                all_min_values.append(spike_waveform.min())
                all_max_values.append(spike_waveform.max())

    if not all_min_values or not all_max_values:
        print("No valid spike data found.")
        return
    global_min = min(all_min_values) - 5
    global_max = max(all_max_values) + 5
    num_dfs = len(filtered_dfs)
    num_cols = 2  # Display 2 plots per row
    num_rows = (num_dfs + 1) // num_cols  # Compute needed rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 2 * num_rows))  # Adjust size based on rows
    axes = axes.flatten()  # Flatten in case of 1D behavior

    n = {}
    for idx, (name, sub_df) in enumerate(filtered_dfs.items()):
        ax = axes[idx]
        unique_values = sub_df[param].unique()

        for value in unique_values:
            subset_df = sub_df[sub_df[param] == value]
            all_spikes = []

            for _, row in subset_df.iterrows():
                spike_key = row['Spike_ID']
                if spike_key in combined_dict_filt:
                    all_spikes.append(np.array(combined_dict_filt[spike_key]))
            if not all_spikes:
                continue  # Skip if no valid spikes found
            n[value] = len(all_spikes)
            all_spikes_array = np.vstack(all_spikes)
            mean_spike = np.median(all_spikes_array, axis=0) if median else np.mean(all_spikes_array, axis=0)     #plotting options
            std_spike = np.std(all_spikes_array, axis=0)
            color = color_map[value]
            ax.plot(mean_spike, label=f"{param}: {value}", linewidth=1.5, color=color)
            ax.fill_between(range(len(mean_spike)), mean_spike - std_spike, mean_spike + std_spike, alpha=0.2, color=color)
  
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('mVs', fontsize=8)
        ax.set_title(f"{name}", fontsize=10)
        handles = [plt.Line2D([0], [0], color=color_map[val], lw=2) for val in unique_values]
        labels = [f"{param}: {val}, n= {n[val]}" for val in unique_values]
        ax.legend(handles, labels, loc='upper right', fontsize=6)
        ax.set_ylim(global_min, global_max)
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def combined_boxplots(param, grouped_dfs_dict, group_metadata_dict=None):
    """
    Generates separate box plots for each parameter, grouped by dataset groups, 
    with ANOVA-based significance annotations.

    - Each plot corresponds to one parameter (e.g., ramp_amp, peak_amp, etc.).
    - Within each plot, each dataset group (from grouped_dfs_dict keys) is on the x-axis.
    - Different colors represent the different values of the metadata (e.g., 'S' vs 'A' for dendriticType).
    - Significance stars are added above respective boxplots based on ANOVA results.

    Parameters:
    - param (str): The metadata category to filter by (e.g., "dendriticType").
    - grouped_dfs_dict (dict): Dictionary where keys are "{param} {group}", and 
                               values are corresponding sub-dataframes.

    Returns:
    None: Displays separate box plots for each parameter.
    """
    sns.set(style="whitegrid")
    columns_to_plot = ['ramp_amp', 'inflection_time', 'inflection_amp', 'peak_amp', 'peak_width', 
                       'peak_sharpness', 'exp_lambda', 'exp_const', 'log_isi', 'r_squared_ramp']   
    filtered_dfs = {key: df for key, df in grouped_dfs_dict.items() if key.startswith(param)}
    if not filtered_dfs:
        print(f"No data frames found for param '{param}'.")
        return
    
    # Create a combined DataFrame for plotting
    combined_df = []
    for key, df in filtered_dfs.items():
        group_name = " ".join(key.split(" ")[1:])  # Extract dataset group name
        df = df.copy()  # Ensure modification safety     
        if param not in df.columns:
            print(f"Warning: '{param}' not found in {key}. Skipping this group.")
            continue
        
        df.loc[:, 'MetadataValue'] = df[param]  # Assign varying metadata column
        df.loc[:, 'Group'] = group_name  # Assign group name (from dictionary key)
        combined_df.append(df)  
    combined_df = pd.concat(combined_df, ignore_index=True)
    

    for column in columns_to_plot:
        if combined_df.empty:
            print(f"[SKIPPING] No data to plot for param: {param}")
            continue
        if combined_df[column].dropna().empty:
            print(f"[SKIPPING] All NaNs in '{column}' for param: {param}")
            continue
            
        plt.figure(figsize=(15, 7))
            
        ax = sns.boxplot(
            x='Group', 
            y=column, 
            hue='MetadataValue',  # Color by different metadata values
            data=combined_df
        )

        plt.title(f'Box Plot of {column} by {param}')
        plt.xlabel(f'Dataset Groups (by {param})')
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.legend(title=f'{param} Value', bbox_to_anchor=(1.05, 1), loc='upper left')

        # **ANOVA for significance testing per dataset group**
        group_positions = {}  # Store x positions for annotation
        for idx, grp in enumerate(combined_df['Group'].unique()):
            subset = combined_df[combined_df['Group'] == grp]

            if len(subset['MetadataValue'].unique()) > 1:  # Ensure multiple groups exist
                values = [subset[column][subset['MetadataValue'] == meta_val] for meta_val in subset['MetadataValue'].unique()]
                f_value, p_value = scipy.stats.f_oneway(*values)
                # Determine significance stars
                significance = ""
                if p_value < 0.0001:
                    significance = "****"
                elif p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                
                group_positions[grp] = (idx, max(subset[column]) +.05*(max(subset[column])-min(subset[column])), significance)

        # **Add annotations for each dataset group**
        for grp, (x_pos, y_pos, stars) in group_positions.items():
            if stars:
                plt.text(x_pos, y_pos, stars, ha='center', va='bottom', fontsize=12, color='red')
        
        #below is fixed metadata labels
        raw_labels = ax.get_xticklabels()
        if group_metadata_dict is not None:
            new_labels = []
            for lbl in raw_labels:
                grp = f'{param} {lbl.get_text()}'
                if group_metadata_dict and grp in group_metadata_dict:
                    new_label = f"{grp}\n{group_metadata_dict[grp]}"
                else:
                    new_label = grp
                new_labels.append(new_label)
    
            ax.set_xticklabels(new_labels, rotation=0, fontsize=9)
    
            plt.tight_layout()
            plt.legend(title=f'{param} Value', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()


def generate_group_metadata_dict(grouped_dfs_dict):
    """
    Generate a metadata label string for each group in grouped_dfs_dict.
    Returns a dictionary mapping group name (as string) to metadata summary.
    """
    metadata_dict = {}
    for key, df in grouped_dfs_dict.items():
        # Assume group name follows "param group_num" format (e.g., 'species 1')
        split_key = key.split(" ")
        if len(split_key) < 2:
            continue
        param, group_name = split_key[0], key

        # Define which metadata columns to use (can be customized)
        columns_to_extract = ['Species', 'Sex', 'brainOrigin', 'SomaLayerLoc', 'dendriticType']
        abbrev_map = {
            'Species': 'Sp',
            'Sex': 'S',
            'brainOrigin': 'bO',
            'SomaLayerLoc': 'SLL',
            'dendriticType': 'dT'
        }

        metadata_parts = []
        for col in columns_to_extract:
            if col in df.columns:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 1:
                    if unique_vals[0] == "Macaca fascicularis":
                        val = "Mf"
                    elif unique_vals[0] == "Macaca mulatta":
                        val = "Mm"
                    else:
                        val = str(unique_vals[0])
                elif len(unique_vals) > 1:
                    val = "Mix"
                else:
                    val = "NA"
                metadata_parts.append(f"\n{abbrev_map[col]}='{val}'")
            else:
                metadata_parts.append(f"{abbrev_map[col]}='NA'")

        metadata_str = ",".join(metadata_parts)
        metadata_dict[group_name] = metadata_str

    return metadata_dict


def generate_plots(sub_dfs_dict, grouping_results_df, combined_dict_filt, waveforms = True, boxplots = True, med = False):
    '''
    Plots all combined boxplots and all waveform grids, output will be long but can edit booleans to print one at a time
    combined_dict_filt is the dictionary that stores all of the spike shapes with keys that should match the Spike_ID
    waveforms plots waveforms when true
    boxplots plots boxplots when true
    med puts the waveform plots in median format when true
    returns nothing
    '''
    varyingMetadata = grouping_results_df['Varying Metadata'].unique()
    for param in varyingMetadata:
        if boxplots:
            group_metadata = generate_group_metadata_dict(sub_dfs_dict)   
            combined_boxplots(param, sub_dfs_dict, group_metadata_dict=group_metadata)  
        if waveforms:
            plot_waveforms_grid(sub_dfs_dict, param, combined_dict_filt, median=med)


