import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import glob

#import metadata file
config_dir = "/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/spikeparam/datasets/spe-1/spe1_helper_modules/"
if config_dir not in sys.path:
    sys.path.append(config_dir)
import config

def compile_experiment_results(folder_path):
    """
    Reads all cluster pickles and maps metadata from config.py.
    """
    # 1. Grab all pkl files in the folder
    search_pattern = os.path.join(folder_path, "*.pkl")
    all_files = glob.glob(search_pattern)
    
    master_list = []

    for file in all_files:
        # Load the individual experiment results
        df = pd.read_pickle(file)
        
        # 2. Extract numeric ID (e.g., 'c1' from 'c1_df_clusters.pkl')
        filename = os.path.basename(file)
        cell_id_str = filename.split('_')[0] 
        cell_id_num = int(cell_id_str.replace('c', ''))
        
        # Add basic identifiers
        df['cell_id'] = cell_id_str
        
        # 3. Map metadata from config.py dictionaries
        # Use .get() to avoid errors if a cell_id is missing in config
        patch_info = config.DICT_PATCH_TYPE.get(cell_id_num, "Unknown, Unknown")
        
        # Split "Juxta, IC" into two distinct columns for your table
        df['patch_type'], df['current_type'] = patch_info.split(', ')
        
        df['cell_type'] = config.DICT_CELL_TYPE.get(cell_id_num)
        df['cortical_depth'] = config.DICT_CORT_DEPTH.get(cell_id_num)
        df['dark_neuron'] = config.DICT_DARK_NEURONS.get(cell_id_num)
        df['clear_EAP_waveform'] = config.DICT_CLEAR_EAP_WAV.get(cell_id_num)
        
        master_list.append(df)

    # 4. Concatenate and Reorder
    final_table = pd.concat(master_list, ignore_index=True)
    
    # Rename for clarity to match your whiteboard
    final_table = final_table.rename(columns={
        'feature_clustered': 'spike_feature',
        'groups': 'cluster'
    })

    # Order columns as requested
    cols = [
        'cell_id', 'spike_feature', 'cluster', 'nRMSE', 'cos_sim',
        'patch_type', 'current_type', 'cell_type', 'cortical_depth', 
        'dark_neuron', 'clear_EAP_waveform'
    ]
    
    return final_table[cols]


def save_clean_merged_table(df, filename='final_merged_report.png', save_fig=False):
    # 1. Prep and Column Ordering
    cols_order = [
        'cell_id', 'patch_type', 'current_type', 'cell_type', 'cortical_depth', 
        'dark_neuron', 'clear_EAP_waveform', 'spike_feature', 'cluster', 'nRMSE', 'cos_sim'
    ]
    plot_data = df[cols_order].copy()
    
    # Format numeric values
    plot_data['nRMSE'] = plot_data['nRMSE'].map(lambda x: f'{x:.3f}' if isinstance(x, float) else x)
    plot_data['cos_sim'] = plot_data['cos_sim'].map(lambda x: f'{x:.3f}' if isinstance(x, float) else x)
    plot_data['cortical_depth'] = plot_data['cortical_depth'].map(lambda x: f'{x:.1f}' if isinstance(x, float) else x)
    
    # 2. Custom Header Formatting
    headers = []
    for col in plot_data.columns:
        if col == 'clear_EAP_waveform':
            headers.append("Clear EAP\nWaveform") 
        elif col == 'nRMSE':
            headers.append("nRMSE") 
        elif col == 'cos_sim':
            headers.append("Cos Sim")
        else:
            headers.append(col.replace('_', ' ').title())

    # 3. Setup Figure
    fig_height = len(plot_data) * 0.6 + 2
    fig, ax = plt.subplots(figsize=(22, fig_height))
    ax.axis('off')

    table = ax.table(
        cellText=plot_data.values,
        colLabels=headers,
        cellLoc='center',
        loc='center'
    )

    # 4. Merging & Centering Logic
    def merge_cells(table, row_start, row_end, col):
        main_cell = table[row_start, col]
        for row in range(row_start + 1, row_end + 1):
            cell = table[row, col]
            cell.get_text().set_text("")
            cell.visible_edges = 'LR' 
        
        table[row_end, col].visible_edges = 'LRB' 
        main_cell.visible_edges = 'LRT'
        main_cell.get_text().set_verticalalignment('center')

    # Apply merges based on Cell ID and Feature blocks
    start_row = 1 
    for i in range(1, len(plot_data) + 1):
        if i == len(plot_data) or plot_data.iloc[i]['cell_id'] != plot_data.iloc[start_row-1]['cell_id']:
            end_row = i
            for c in range(7): # Metadata
                merge_cells(table, start_row, end_row, c)
            
            feat_start = start_row
            for j in range(start_row, end_row + 1):
                current_feat = plot_data.iloc[j-1]['spike_feature']
                next_feat = plot_data.iloc[j]['spike_feature'] if j < end_row else None
                if next_feat != current_feat:
                    merge_cells(table, feat_start, j, 7)
                    feat_start = j + 1
            start_row = i + 1

    # 5. Final Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3.5) 

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#BDBDBD')
        cell.get_text().set_verticalalignment('center')
        
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
            cell.visible_edges = 'closed'
        else:
            if col == 0: # BOLD the Cell ID column
                cell.set_text_props(weight='bold')
            if col >= 9:
                cell.set_facecolor('#F8F9FA')

    # 6. Optional Saving Logic
    if save_fig:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Figure saved as {filename}")
    
    plt.show()
