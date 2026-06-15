import os
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from neurodsp import spectral
import matplotlib.pyplot as plt


def _classify_stimulus(stim_data):
    """Return the historical stimulus label using the existing correlation rule."""

    stim_nonzero = stim_data.copy()
    stim_nonzero[0:5000] = 0
    stim_nonzero = stim_nonzero[stim_nonzero != 0]

    if np.size(stim_nonzero) <= 2:
        return 'none', stim_nonzero, np.nan

    r, _ = pearsonr(np.arange(np.size(stim_nonzero)), stim_nonzero)
    if np.isnan(r):
        return 'constant', stim_nonzero, r
    if r > 0.95:
        return 'ramp', stim_nonzero, r
    return 'pink', stim_nonzero, r



# load sweeps
def load_sweep(sweep_number, f, fs):
    # load sweep
    dset = f['Sweep_' + str(sweep_number)]

    # create array of time indices
    convert_to_ms = (fs / 1000)
    times = np.arange(0, np.shape(dset)[0]) / convert_to_ms # in ms instead of sec

    return dset, times

# get the power spectrum of the stimulus
def get_stim_spectrum(dset, fs, nperseg, all_data=True):
    
    if all_data == True:
        stim_data = dset[:, 0]
        fxx, pxx = spectral.compute_spectrum(stim_data, fs, method='welch',
                    window='hann', nperseg=nperseg)
    
    if all_data == False:
        fxx, pxx = spectral.compute_spectrum(dset, fs, method='welch',
                window='hann', nperseg=nperseg)
        
    return fxx, pxx
    
# plot the ephys data
def plot_ephys_data(dset, times, index_start, index_end):
    plot_range = (index_start, index_end)
    plt.plot(times[plot_range[0]:plot_range[1]],
        dset[plot_range[0]:plot_range[1], 1], '.')
        
# plot the stimulus
def plot_stim_data(dset, times, index_start, index_end):
    plot_range = (index_start, index_end)
    plt.plot(times[plot_range[0]:plot_range[1]],
        dset[plot_range[0]:plot_range[1], 0], '.')

# plot the stimulus spectrum
def plot_stim_spectrum(fxx, pxx, xlim, ylim):
    plt.loglog(fxx, pxx)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (V^2/Hz)')
    plt.xlim(xlim)
    plt.ylim(ylim)

# find the spike times
def find_spike_times(data, thresh_mv, thresh_ms):
    # thresh_mv is the voltage threshold that needs to be crossed
    # thresh_ms is how many ms each peak needs to be from the next

    peaks = find_peaks(data, height=thresh_mv, distance=thresh_ms)
    idx_spikes = peaks[0] # spike indices
    amp_spikes = peaks[1]['peak_heights'] # spike amplitudes

    return idx_spikes, amp_spikes

# define exponential function
def exp_func(times, exp_amp, exp_lambda, exp_timeshift, exp_const):
    return exp_amp * np.exp(-exp_lambda * (times - exp_timeshift)) + exp_const

# fit exponential data
def fit_exp_nonlinear(times, data, initial_guesses, bounds):
    popt, _ = curve_fit(exp_func, times, data, p0=initial_guesses, bounds=bounds,  maxfev=10000)
    exp_amp, exp_lambda, exp_timeshift, exp_const = popt
    return exp_amp, exp_lambda, exp_timeshift, exp_const
    
    

def process_pink_type_info(f, fs):
    #generate dict with info of pink noise variations
    all_pink_noise = {}
    pink_type_num = 0
    #pink_stim_dict = {}
    pink_types = []
    
    
    
    # 0 through 66 sweeps
    for i_sweeps in range(66):
    
    
        #load data
        dset, times = load_sweep(i_sweeps, f, fs)
    
        stim_type, stim_data, r = _classify_stimulus(dset[:, 0])
        pink_type = stim_type
        if stim_type == 'pink':
            plt.plot(stim_data)
    
    
        if stim_type == 'pink':
            
            #get specific variation of pink noise 
            if r  not in all_pink_noise:
                pink_type_num = pink_type_num + 1
                pink_type = 'PinkNoise'+str(pink_type_num)
                all_pink_noise[r] = pink_type
                #get power spec for that pink noise type 
                fxx, pxx = get_stim_spectrum(stim_data, fs, nperseg=fs/4, all_data = False)
                xlim = (10e-2, 10e5)
                ylim = (10e-10, 10e-1)
        

            else:
                pink_type = all_pink_noise[r]
            
        pink_types.append(pink_type)

    return pink_types


def process_sweeps(num_sweeps,f, one_ms, fs, pink_types, load_sweep, find_spike_times, fit_exp_nonlinear):
    """Processes multiple sweeps, extracting spike features and stimulus characteristics."""

    loaded_sweeps = [load_sweep(i_sweeps, f, fs) for i_sweeps in range(66)]
    all_data = []
    all_times = []
    for dset, times in loaded_sweeps:
        data = dset[:, 1] # ephys data
        all_times.append(times)
        all_data.append(data)

    # Initialize lists for storing results
    df_sweep = []
    df_stim_type = []
    df_spike_number = []
    df_voltage_ramp = []
    df_inflection_time = []
    df_inflection_mv = []
    df_peak_amplitude = []
    df_peak_sharpness = []
    df_decay_lambda = []
    df_decay_const = []
    df_stim_exp = []
    df_stim_mean = []
    df_stim_std = []
    df_pinktype = []

    all_contant_spks = []
    all_ramp_spks = []
    all_pink_spks = []

    for i_sweeps in range(num_sweeps):
        dset, times = loaded_sweeps[i_sweeps]
        stim = dset[:, 0]  # current injection
        data = dset[:, 1]  # ephys data

        # Find spikes
        thresh_mv = -10
        thresh_ms = one_ms * 1
        idx_spikes, amp_spikes = find_spike_times(data, thresh_mv, thresh_ms)

        # Identify stimulus type
        stim_type, stim_data, _ = _classify_stimulus(stim)

        # Process each detected spike
        for i_spikes in range(np.size(idx_spikes)):
            window_length = (10, 10)
            window_pre = int(one_ms * window_length[0])
            window_post = int(one_ms * window_length[1])

            window_spike_pre = idx_spikes[i_spikes] - window_pre
            window_spike_post = idx_spikes[i_spikes] + window_post

            windowed_times = times[window_spike_pre:window_spike_post]
            windowed_data = data[window_spike_pre:window_spike_post]

            d_windowed_data = np.diff(windowed_data)
            smoothed_data = sm.nonparametric.lowess(d_windowed_data, windowed_times[1:], frac=0.008)
            d_smoothed_data = smoothed_data[:, 1]

            if stim_type == 'constant':
                all_contant_spks.append(d_smoothed_data)
            elif stim_type == 'ramp':
                all_ramp_spks.append(d_smoothed_data)
            elif stim_type == 'pink':
                all_pink_spks.append(d_smoothed_data)

            # Compute inflection point
            noise_window = d_smoothed_data[:one_ms]
            noise_mean = np.mean(noise_window)
            noise_std = np.std(noise_window)
            z_data = (d_smoothed_data - noise_mean) / noise_std

            idx_z_peak, _ = find_spike_times(z_data, 40, thresh_ms)
            idx_spike_peak, spike_peak_amp = find_spike_times(windowed_data, thresh_mv, thresh_ms)

            inflection_time = np.abs(z_data[:idx_z_peak[0]] - 40)
            idx_inflection = np.argmin(inflection_time)

            idx_ramp_start = idx_inflection - int(0.5 * one_ms)
            ramp_times = windowed_times[idx_ramp_start:idx_inflection]
            voltage_ramp, _ = np.polyfit(ramp_times, windowed_data[idx_ramp_start:idx_inflection], 1)

            inflection_time = idx_spike_peak - idx_inflection
            inflection_time = inflection_time[0] / one_ms
            inflection_mv = windowed_data[idx_inflection]

            # Peak voltage and sharpness
            mv_peak = spike_peak_amp[0]
            sharpness_peak = ((windowed_data[idx_spike_peak[0]] - windowed_data[idx_spike_peak[0] - 5]) +
                              (windowed_data[idx_spike_peak[0]] - windowed_data[idx_spike_peak[0] + 5])) / 2

            # Voltage decay rate
            decay_curve_shift = one_ms / 2
            decay_curve_floor_time = one_ms * 5
            exp_window = idx_spike_peak[0] + int(decay_curve_shift)
            exp_window_end = exp_window + int(decay_curve_floor_time)
            exp_data = windowed_data[exp_window:exp_window_end]
            exp_times = windowed_times[exp_window:exp_window_end]

            initial_guesses = np.array([50, 1, windowed_times[exp_window], -60], dtype=np.float64)
            bounds = ([0, 0, 0, -100], [1000, 3, 10e9, 50])

            exp_amp, exp_lambda, exp_timeshift, exp_const = fit_exp_nonlinear(exp_times, exp_data, initial_guesses, bounds)

            # Stimulus parameters for pink noise
            if stim_type == 'pink':
                inflection_pre_window = int(idx_inflection - (one_ms * 5))
                windowed_stim = stim[window_spike_pre:window_spike_post]
                windowed_stim = windowed_stim[inflection_pre_window:idx_inflection]

                fxx, pxx = spectral.compute_spectrum(windowed_stim, fs, method='welch',
                                                     window='hann', nperseg=np.size(windowed_stim))
                fxx, pxx = fxx[1:], pxx[1:]

                stim_exp, _ = np.polyfit(np.log10(fxx), np.log10(pxx), 1)
                stim_exp = -stim_exp
                stim_mean = np.mean(windowed_stim)
                stim_std = np.std(windowed_stim)
            else:
                stim_exp = np.nan
                stim_mean = np.nan
                stim_std = np.nan

            # Collect extracted features
            df_sweep.append(i_sweeps)
            df_stim_type.append(stim_type)
            df_spike_number.append(i_spikes)
            df_voltage_ramp.append(voltage_ramp)
            df_inflection_time.append(inflection_time)
            df_inflection_mv.append(inflection_mv)
            df_peak_amplitude.append(mv_peak)
            df_peak_sharpness.append(sharpness_peak)
            df_decay_lambda.append(exp_lambda)
            df_decay_const.append(exp_const)
            df_stim_exp.append(stim_exp)
            df_stim_mean.append(stim_mean)
            df_stim_std.append(stim_std)
            df_pinktype.append(pink_types[i_sweeps])

    return df_sweep, df_stim_type, df_spike_number, df_voltage_ramp, df_inflection_time, df_inflection_mv, df_peak_amplitude, df_peak_sharpness, df_decay_lambda, df_decay_const, df_stim_exp, df_stim_mean, df_stim_std, df_pinktype, all_data, all_times, all_contant_spks, all_ramp_spks, all_pink_spks


def plot_spike_and_derivative(i_sweeps, f, fs, one_ms):
    dset, times = load_sweep(i_sweeps, f, fs)

    stim = dset[:, 0] # current injection
    data = dset[:, 1] # ephys data
    
    #########################
    # find spikes
    thresh_mv = -10
    thresh_ms = one_ms * 1 # 1 ms
    
    idx_spikes, amp_spikes = find_spike_times(data, thresh_mv, thresh_ms)
    
    #########################
    i_spikes = 0
        
    # get windows around spikes
    # create window indices
    window_length = (10, 10) # in ms
    window_pre = int(one_ms * window_length[0])
    window_post = int(one_ms * window_length[1])
    
    # get window
    window_spike_pre = (idx_spikes[i_spikes]-window_pre)
    window_spike_post = (idx_spikes[i_spikes]+window_post)
    
    # get window for times as well
    windowed_times = times[(idx_spikes[i_spikes]-window_pre):
                        (idx_spikes[i_spikes]+window_post)]
    
    # get data window
    windowed_data = data[window_spike_pre:window_spike_post]
    
    # difference of data, and smooth it
    d_windowed_data = np.diff(windowed_data)
    smoothed_data = sm.nonparametric.lowess(d_windowed_data, windowed_times[1:], frac=0.008)
    
    # smoothed data
    d_smoothed_times = smoothed_data[:, 0]
    d_smoothed_data = smoothed_data[:, 1]
    
    plt.plot(windowed_times, windowed_data, 'b', alpha = 0.8, linewidth = 3., label='intra')
    plt.plot(d_smoothed_times, (d_smoothed_data*80)-50, 'k', linewidth = 3., label='differenced')
    plt.legend()
    plt.show()


def load_or_compute_cell_data(pickle_paths, force_rerun,
                              n_sweeps, f, one_ms, fs, pink_types):
    """Load df_stim_features and all_data_flat from cache, or compute from scratch.

    pickle_paths must have keys 'stim' and 'data'. all_data_flat is always computed
    on the fly from all_data (not pickled — it is too large to save efficiently).
    Returns (df_stim_features, all_data_flat).
    """
    import pickle, pandas as pd

    all_data = None
    if not force_rerun and os.path.exists(pickle_paths['stim']) and os.path.exists(pickle_paths['data']):
        try:
            with open(pickle_paths['stim'], 'rb') as fh:
                df_stim_features = pickle.load(fh)
            with open(pickle_paths['data'], 'rb') as fh:
                all_data = pickle.load(fh)
            print('Loaded df_stim_features and all_data from pickle.')
        except Exception as e:
            print(f'Pickle load failed ({e}); recomputing.')

    if all_data is None:
        (df_sweep, df_stim_type, df_spike_number, df_voltage_ramp, df_inflection_time, df_inflection_mv,
         df_peak_amplitude, df_peak_sharpness, df_decay_lambda, df_decay_const, df_stim_exp, df_stim_mean,
         df_stim_std, df_pinktype, all_data, all_times, *_
        ) = process_sweeps(n_sweeps, f, one_ms, fs, pink_types,
                           load_sweep, find_spike_times, fit_exp_nonlinear)
        df_stim_features = pd.DataFrame({
            'sweep': df_sweep, 'stim_type': df_stim_type, 'spike_num': df_spike_number,
            'stim_exp': df_stim_exp, 'stim_mean': df_stim_mean,
            'stim_std': df_stim_std, 'pink_type': df_pinktype,
        }, columns=['sweep', 'stim_type', 'spike_num', 'stim_exp', 'stim_mean', 'stim_std', 'pink_type'])
        with open(pickle_paths['stim'], 'wb') as fh:
            pickle.dump(df_stim_features, fh)
        with open(pickle_paths['data'], 'wb') as fh:
            pickle.dump(all_data, fh)
        print('Computed and saved df_stim_features and all_data to pickle.')

    all_data_flat = [spike for sweep in all_data for spike in sweep]
    return df_stim_features, all_data_flat
