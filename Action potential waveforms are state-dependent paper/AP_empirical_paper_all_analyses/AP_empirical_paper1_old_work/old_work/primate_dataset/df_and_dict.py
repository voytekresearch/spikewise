'''
Author: David Brin
Date Rewritten: 2/12/2025

This python file is to recreate the data frame containing all the spikes and their features 
along with the dictionary containing all the spike waveforms that will correspond to the spikes in the data frame via spike ID. 
'''
#imports
import sys
sys.path.append(r'..\..\..\spikewise')
from spikewise.patch.fit import Spike
from spikewise.patch.fit import SpikeGroup
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
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy.spatial as sp_spatial
import os
from fooof import FOOOF
sns.set(rc={'figure.figsize':(12,9)})
sns.set_style('whitegrid')
sns.set_style("whitegrid", {'axes.grid' : False})
import IProgress
import openpyxl
import pickle


# Function to visualize sweep patch data
# no metadata, just the time series data
def extract_data(file_path, plot_data = False):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Initialize an empty list to store data arrays
        data = []

        # Iterate through keys in the 'acquisition' group
        for sweep_key in f['acquisition'].keys():
            dataset = f['acquisition'][sweep_key]['data'] 
            # Convert the dataset data into a NumPy array and append to the list
            data.append(np.array(dataset))

        # Plot the data
        if(plot_data):
            if all(d.ndim == 1 for d in data):
                for d in data:
                    plt.plot(d)
                plt.xlabel('time (ms)')
                plt.ylabel('mV')
                plt.title('1D Dataset Visualization')
                plt.show()
            elif all(d.ndim == 2 for d in data):
                for d in data:
                    plt.imshow(d, cmap='viridis')
                    plt.colorbar()
                    plt.xlabel('X-axis')
                    plt.ylabel('Y-axis')
                    plt.title('2D Dataset Visualization')
                    plt.show()
            else:
                print("Cannot visualize data with more than 2 dimensions.")



        return data

#collecting file paths

def get_file_paths(folder_path):
    """
    Function to loop through a folder and save file paths.
    
    Args:
    - folder_path (str): Path to the folder to loop through.
    
    Returns:
    - file_paths (list): List of file paths found in the folder.
    """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths 
def update_columns_at_index(df, file_path):
    """
    Function to update columns in the DataFrame at a specific index with values from Excel metadata.
    
    Check if filename matches the string in the first cell of the row, drop row if not.

    Args:
    - df (pd.DataFrame): DataFrame to update.
    - file_path (str): Path to the Excel file containing metadata.
    
    Returns:
    - true if updated
    """
    # Extract filename from file_path
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Load the workbook
        #wb = openpyxl.load_workbook(r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\ephys_features_filenames (1).xlsx")
    metadf = pd.read_excel(r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\ephys_features_filenames (1).xlsx")
        # Select the active worksheet
        #ws = wb.active
    

    # Filter the metadata DataFrame to find the row associated with this file
    file_metadata = metadf[metadf['file_name'] == filename]
    if not file_metadata.empty:    
        for col in file_metadata.columns:
            df[col] = file_metadata[col].values[0]  # Assign metadata value to the entire column
            return True
    
    else:
        print(f"No metadata found for file: {filename}")
        return False
def monkey_df_and_dict(filepaths):
    """
    Creates a DataFrame and dictionary of spike data for all files in the given filepaths (Monkey Directoru.

    Parameters:
    - filepaths: List of file paths to process.
    - ind_start: Starting index for the metadata cols for 'update_columns_at_index' for the monkey data frame.

    Returns:
    - super_mega_df: A concatenated DataFrame of spike features for all files.
    """
    super_mega_df = pd.DataFrame()
    monkey_dict = {}  # Monkey dictionary
    file_num = 0

    for file in filepaths:
        print(file)
        # Extract data for all sweeps in the file
        data = extract_data(file, plot_data=False)  # Numpy array of all sweeps in the file
        spike_dir = {}  # Dictionary of sweeps with spikes
        i = 0
        monkey_id = os.path.basename(os.path.dirname(file))  # Extract Monkey_ID from the directory name
        fileName = os.path.splitext(os.path.basename(file))[0] 
        with h5py.File(file, 'r') as f:
            for sweep_key in f['acquisition'].keys():
                if i < len(data):
                    try:
                        # Fit spike data
                        sweep_key_obj = Spike(thresh_amp=0, window_length=(5., 5.), smooth_frac=.01)
                        sweep_key_obj.fit(data[i], 20000, n_jobs=-1, progress=tqdm)
                        if sweep_key_obj.n_spikes is not None:
                            spike_dir[sweep_key] = sweep_key_obj
                            #print(f'num spikes: {sweep_key_obj.n_spikes} and len df: {len(sweep_key_obj.df_features)}')
                            #print(f"Length of sweep_key_obj.spikes: {len(sweep_key_obj.spikes)}")
                    except ValueError as e:
                        print(f"Fitting failed for sweep {sweep_key}: {e}")
                i += 1

        # Create and concatenate DataFrame with all sweeps
        mega_df = pd.DataFrame()
        mega_dict = {}             #store all spikes in file and concatenate if file appears in metadata spreadsheet
        for i, (sweep_key, sweep_obj) in enumerate(spike_dir.items()):
            print(sweep_key, sweep_obj)
            df = sweep_obj.df_features.copy()  # Copy DataFrame to avoid modifying the original
            
            df['Sweep_#'] = sweep_key
            df['Spike_#'] = range(0, len(df))  # Add Spike_# as a column
            
            # Create Spike_IDs in the desired format
            df['Spike_ID'] = df.apply(
                lambda row: f"{monkey_id}f{file_num}Sw{sweep_key}Sp{row['Spike_#']}", axis=1
            )
            #display(df)
            for spike in range(len(df)):
                key = f"{monkey_id}f{file_num}Sw{sweep_key}Sp{spike}"                      
                mega_dict[key] = sweep_obj.spikes[spike]
            # Concatenate the current DataFrame to the mega DataFrame
            
            mega_df = pd.concat([mega_df, df], axis=0)
            #print(f'num spikes: {sweep_obj.n_spikes} and len df: {len(sweep_obj.df_features)}')

        # Update columns and append to the super_mega_df
        exists = update_columns_at_index(mega_df, file)
        if exists:
            super_mega_df = pd.concat([super_mega_df, mega_df], axis=0)
            monkey_dict.update(mega_dict)
            print("File added")
            print(f"\n\n\n\ndf length: {len(super_mega_df)} \n dict length: {len(monkey_dict.keys())} \n\n\n\n")
        
        file_num += 1

    return super_mega_df, monkey_dict
def create_allMonkey_data(file_paths):
    '''
    cohesive function that calls monkey_df_and_dict on all monkey folders, puts all data into allMonkey_df and combined_dict
    couldn't make it fully adaptable because of 'update_columns_at_index'
    
    returns allMonkey_df, combined_dict
    '''
    allMonkey_df = pd.DataFrame()
    combined_dict = {}
    print("Creating data frame and dictionary")
    for file in file_paths:
        file_list = [os.path.join(file, f) for f in os.listdir(file) if os.path.isfile(os.path.join(file, f))]
        monk_df,monk_dict = monkey_df_and_dict(file_list)     
        if not monk_df.empty:
            allMonkey_df = pd.concat([allMonkey_df, monk_df], axis=0, ignore_index=True)
        else:
            print(f"\n\n\n\n\nWarning: No data found in {file}\n\n\n\n\n\n\n\n\n") #run a check in case empty file
        combined_dict.update(monk_dict)
        print(f"\n\n\n\n\n\n\nconcat df length: {len(allMonkey_df)} \nconcat dict length: {len(combined_dict.keys())} \n\n\n\n\n\n\n")
    return allMonkey_df, combined_dict

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

'''
To run:
file_paths = [r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\M03", r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\M04", r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\M05"
             ,r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\M06", r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\M08"
             , r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\M10", r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\M11"
             , r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\M12", r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\M19"
             , r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\M20", r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\M21"]

allMonkey_df, combined_dict = create_allMonkey_data(file_paths)
save_dataframe_to_pickle(allMonkey_df, r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\allMonkey_df.pkl")
save_dict_to_pickle(combined_dict, r"C:\Users\david\Documents\Voytek Research\spike_proj\primate Dataset\combined_dict.pkl")
'''