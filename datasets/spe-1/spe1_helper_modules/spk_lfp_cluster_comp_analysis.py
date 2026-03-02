import os
import glob
import pickle
import pandas as pd




def compile_lfp_stats(pickle_dir="/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/spe1_pickles/lfp_spk_group_pickles"):
    all_files = glob.glob(os.path.join(pickle_dir, "*_sliding_stats.pkl"))
    
    master_stats_rows = []
    master_traces = []
    
    for file_path in all_files:
        with open(file_path, 'rb') as f:
            cell_data = pickle.load(f)
            
        cell_id = cell_data.get("cell_id", "Unknown")
        
        for feature_name, data in cell_data.items():
            if feature_name == "cell_id": continue 
                
            # --- 1. EXTRACT STATS ---
            for pair in data.get("pairwise_stats", []):
                g1 = pair['group_1'].split(': ')[-1] if ':' in pair['group_1'] else pair['group_1']
                g2 = pair['group_2'].split(': ')[-1] if ':' in pair['group_2'] else pair['group_2']
                
                master_stats_rows.append({
                    "cell_id": cell_id,
                    "spike_feature": pair.get("spike_feature", "Unknown"), # Pulled from inside the stats!
                    "lfp_feature": feature_name,
                    "window_start": pair["window_start"],
                    "window_end": pair["window_end"],
                    "group_1": g1,
                    "group_2": g2,
                    "comparison": f"{g1} vs {g2}",
                    "p_value": pair["p_value"],
                    "cohens_d": pair["cohens_d"]
                })
                
            # --- 2. EXTRACT TRACES ---
            for spike_feature, t_data in data.get("trace_data", {}).items():
                master_traces.append({
                    "cell_id": cell_id,
                    "spike_feature": spike_feature,
                    "lfp_feature": feature_name,
                    "trace_data": t_data
                })
                
    return pd.DataFrame(master_stats_rows), master_traces
