"""Perform spectral decomposition and parameterization of MEG data."""

# Import necessary modules
import os
import sys
import time
import warnings
import mne
import ray
import numpy as np
import fooof
from fooof.utils import trim_spectrum
import pandas as pd
import params


@ray.remote
def run_decomp_and_sparam_one_trial(
    trial,
    trial_num,
    fmin=params.FMIN,
    fmax=params.FMAX,
    n_freqs=params.N_FREQS,
    time_window_len=params.TIME_WINDOW_LEN,
    decim_factor=params.DECIM_FACTOR,
    n_peaks=params.N_PEAKS,
    peak_width_lims=params.PEAK_WIDTH_LIMS,
    freq_bands=params.FREQ_BANDS,
    verbose=True,
):
    """
    For one trial of data, run spectral decomposition and spectral
    parameterization.

    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs object to run spectral decomposition and spectral
        parameterization.
    trial_num : int
        Trial number to run spectral decomposition and spectral
        parameterization for.
    fmin : float (default: params.FMIN)
        Minimum frequency to use for spectral decomposition.
    fmax : float (default: params.FMAX)
        Maximum frequency to use for spectral decomposition.
    n_freqs : int (default: params.N_FREQS)
        Number of frequencies to use for spectral decomposition.
    time_window_len : float (default: params.TIME_WINDOW_LEN)
        Length of time window to use for spectral decomposition.
    decim_factor : int (default: params.DECIM_FACTOR)
        Decimation factor to use for spectral decomposition.
    n_peaks : int (default: params.N_PEAKS)
        Maximum number of peaks to use for spectral parameterization.
    peak_width_lims : tuple (default: params.PEAK_WIDTH_LIMS)
        Peak width limits to use for spectral parameterization.
    freq_bands : tuple (default: params.ALPHA_BAND)
        Frequency band to extract peaks from spectral parameterization.
    verbose : bool (default: True)
        Whether to print runtime information.
    """
    # Make frequencies linearly spaced
    freqs = np.linspace(fmin, fmax, n_freqs)

    # Make time window length consistent across frequencies
    n_cycles = freqs * time_window_len

    # Use multiple tapers to estimate spectrogram
    trial_tfr = mne.time_frequency.tfr_multitaper(
        trial,
        freqs,
        n_cycles,
        return_itc=False,
        picks="meg",
        average=False,
        decim=decim_factor,
        verbose=False,
    )

    # Reshape spectrogram to be (n_channels, n_timepts, n_freqs)
    tfr_arr = np.squeeze(np.swapaxes(trial_tfr.data, 2, 3))

    # Start timer for spectral parameterization
    start = time.time()

    # Initialize FOOOFGroup
    fooof_grp = fooof.FOOOFGroup(
        max_n_peaks=n_peaks, peak_width_limits=peak_width_lims, verbose=False
    )

    with warnings.catch_warnings():
        # Ignore warnings for bad fits
        warnings.simplefilter("ignore")

        # Fit spectral parameterization model
        fooof_grp.fit(
            freqs, tfr_arr.reshape(-1, len(freqs)), freq_range=(fmin, fmax)
        )

    # Make DataFrame for spectral parameterization results
    model_df = fooof_grp.to_df(fooof.Bands(freq_bands))

    # Initialize list of fitted models and area parameters
    n_channels, n_timepts, n_freqs = tfr_arr.shape
    area_params_dct = {
        "logOscAUC": None,
        "logTotAUC": None,
        "linOscAUC": None,
        "linTotAUC": None,
    }
    area_params = np.zeros(
        (n_channels * n_timepts, len(freq_bands) * len(area_params_dct))
    )

    # Fit spectral parameterization model for one channel and time point
    for i in range(len(fooof_grp.group_results)):
        # Regenerate FOOOF model
        fm = fooof_grp.get_fooof(i)
        if not fm.has_model or not fm.has_data:
            print(0, fm.has_model, fm.has_data)
            continue
        # Determine all areas to extract
        area_params_dct["logOscAUC"] = fm.power_spectrum - fm._ap_fit
        area_params_dct["logTotAUC"] = fm.power_spectrum
        area_params_dct["linOscAUC"] = 10**fm.power_spectrum - 10**fm._ap_fit
        area_params_dct["linTotAUC"] = 10**fm.power_spectrum
        area_params_one_psd = {}
        for param, spectra in area_params_dct.items():
            for tag, band in freq_bands.items():
                freqs_trim, psd_trim = trim_spectrum(freqs, spectra, band)
                area_params_one_psd[f"{tag}_{param}"] = np.trapz(
                    psd_trim, freqs_trim
                )

        # Add values to area parameters array
        area_params[i] = np.array(
            [
                area_params_one_psd[k]
                for k in sorted(area_params_one_psd.keys())
            ]
        )

    # Make DataFrame for spectral parameterization results
    area_params_df = pd.DataFrame(
        area_params, columns=sorted(area_params_one_psd.keys())
    )

    # Join model and area parameter DataFrames and add trial, channel, and time
    # point
    sparam_df_one_trial = model_df.join(area_params_df)
    index_shape = (1, n_channels, n_timepts)
    index_names = ["trial", "channel", "timepoint"]
    index = pd.MultiIndex.from_product(
        [range(s) for s in index_shape], names=index_names
    )
    sparam_df_one_trial.index = index
    sparam_df_one_trial = sparam_df_one_trial.reset_index()

    # Print runtime for spectral parameterization
    if verbose:
        print(
            f"Finished spectral parameterization for trial #{trial_num} in "
            f"{time.time() - start:.1f} seconds"
        )
    return trial_num, sparam_df_one_trial


def run_decomp_and_sparam_all_trials(
    epochs,
    save_dir,
    verbose=True,
):
    """For each trial of data, run spectral decomposition and spectral
    parameterization using ray for parallelization.

    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs object containing data to be parameterized.
    save_dir : str
        Directory to save results to.

    Returns:
    --------
    sparam_df : pd.DataFrame
        DataFrame containing spectral parameterization results for all trials.
    """
    # Start timer for spectral decomposition and parameterization
    start = time.time()

    # Determine which trials have already been computed
    trials_computed = [
        int(f.split(".")[0].split("l")[-1]) for f in os.listdir(save_dir)
    ]

    # Determine number of trials
    n_trials = epochs.get_data(copy=True).shape[0]

    # If not all trials have been computed, compute remaining trials
    if not len(trials_computed) == n_trials:
        # Print remaining trials to compute
        trials_to_sparam = sorted(
            list(set(range(n_trials)) - set(trials_computed))
        )
        if verbose:
            print(
                f"Already parameterized: {len(trials_computed)}\n"
                f"Still to parameterize: {len(trials_to_sparam)}\n"
            )

        # Iterate through each trial of data
        ray.init()

        # Fit spectral parameterization model for one trial
        result_ids = [
            run_decomp_and_sparam_one_trial.remote(
                epochs[trial_num], trial_num
            )
            for trial_num in trials_to_sparam
        ]

        # Save trial data as it becomes available
        while result_ids:
            done_id, result_ids = ray.wait(result_ids)

            # Save trial DataFrame
            trial_num, sparam_df_one_trial = ray.get(done_id[0])
            save_fname = f"{save_dir}/sparam_trial{trial_num}.csv"
            sparam_df_one_trial.to_csv(save_fname, index=False)

    # Concatenate all trial DataFrames
    sparam_df_all_trials, skipped_lst = pd.DataFrame([]), []
    for fname in os.listdir(save_dir):
        try:
            sparam_df_one_trial = pd.read_csv(f"{save_dir}/{fname}")
        except pd.errors.EmptyDataError:
            skipped_lst.append(int(fname.split(".")[0].split("l")[-1]))
        sparam_df_all_trials = pd.concat(
            [sparam_df_all_trials, sparam_df_one_trial], ignore_index=True
        )

    # Print runtime for spectral decomposition and parameterization
    if verbose:
        print(
            f"\nFinished spectral decomposition and parameterization for "
            f"{n_trials} trials in {time.time() - start:.1f} seconds"
        )
    return sparam_df_all_trials, skipped_lst


def convert_sparam_df_to_mne(sparam_df, info, save_fname, verbose=True):
    """Convert spectral parameterization DataFrame to series of MNE epochs, one
    for each relevant spectral parameterization model parameter.

    Parameters:
    -----------
    sparam_df : pd.DataFrame
        DataFrame containing spectral parameterization results for all trials.
    info : mne.Info
        Info object containing metadata for MNE epochs.
    save_fname : str
        Filename to save spectral parameterization results to.  This is used to
        determine the filename for the MNE epochs.
    """
    # Start timer for conversion to MNE
    start = time.time()

    # Reorganize spectral parameterization DataFrame
    sparam_df = sparam_df.set_index(["trial", "channel", "timepoint"])
    sparam_df = sparam_df.sort_index()
    if len(sparam_df) % np.prod(sparam_df.index.levshape[1:]):
        sparam_df = sparam_df.reindex(
            pd.MultiIndex.from_product(
                [np.arange(n) for n in sparam_df.index.levshape],
                names=["trial", "channel", "timepoint"],
            )
        )

    # Fill remaining NaNs with zero
    sparam_df = sparam_df.fillna(0)

    # Iterate through each model parameter
    for col in sparam_df.columns:
        # Determine filename for selected model parameter
        col_epochs_fname = save_fname.replace("_epo", f"_{col}_epo")

        # Avoid recomputing
        if os.path.exists(col_epochs_fname):
            continue

        # Make MNE Epochs for selected model parameter
        arr = sparam_df[col].values.reshape(-1, *sparam_df.index.levshape[1:])
        epochs_arr = mne.EpochsArray(arr, info, verbose=False)

        # Save EpochArray
        epochs_arr.save(col_epochs_fname)

    # Print runtime for conversion to MNE
    if verbose:
        print(
            f"\nConverted spectral parameterization results to MNE "
            f"Epochs in {time.time() - start:.1f} seconds"
        )


def spec_decomp_and_param_one_subject(
    epochs_fname,
    sparam_dir=params.SPARAM_DIR,
):
    """Load data and then perform spectral decomposition and parameterization
    for one subject."""
    # Make directory to save data to if necessary
    os.makedirs(sparam_dir, exist_ok=True)

    # Get subject name
    subject_id = epochs_fname.split("/")[-1].split("-")[0].split("_")[-1]

    # Determine whether subject has already been processed
    subject_fifs = [
        f
        for f in os.listdir(sparam_dir)
        if f.startswith(f"{subject_id}_") and f.endswith(".fif")
    ]

    # See if each of 15 parameters have been computed
    # (offset, exponent, CF, PW, BW, R^2, mse) + area parameters
    if len(subject_fifs) == 15:
        return

    # Print subject info
    print(f"\nProcessing Subject {subject_id}")

    # Load subject's MEG data
    epochs = mne.read_epochs(epochs_fname, verbose=False)
    print(epochs.get_data().shape)

    # Compute and parameterize spectrogram
    subj_sparam_dir = f"{sparam_dir}/{subject_id}"
    os.makedirs(subj_sparam_dir, exist_ok=True)
    print(
        f"\nStarting spectral decomposition and parameterization for Subject "
        f"{subject_id}"
    )
    sparam_df, skipped_lst = run_decomp_and_sparam_all_trials(
        epochs, subj_sparam_dir
    )

    # Log skipped trials
    skipped_dct, skipped_fname = {}, f"{sparam_dir}/skipped.csv"
    if len(skipped_lst) > 0:
        if os.path.exists(skipped_fname):
            skipped_df = pd.read_csv(skipped_fname)
            skipped_dct = skipped_df.to_dict(orient="list")
        skipped_dct[subject_id] = skipped_lst
        skipped_df = pd.DataFrame.from_dict(
            skipped_dct, orient="index"
        ).transpose()
        skipped_df.to_csv(skipped_fname, index=False)

    # Extract spectral parameters from model and convert to mne
    sparam_epo_fname = f"{sparam_dir}/{subject_id}_epo.fif"
    convert_sparam_df_to_mne(sparam_df, epochs.info, sparam_epo_fname)


def spec_decomp_and_param_all_subjects(
    subjects_lst=None,
    preprocessed_dir=params.PREPROCESSED_DIR,
):
    """Load data and then perform spectral decomposition and parameterization
    for all subjects.

    Parameters:
    -----------
    preprocessed_dir : str (default: params.PREPROCESSED_DIR)
        Directory to save processed data to.
    """
    # Get list of preprocessed data files
    epochs_fnames = [
        f"{preprocessed_dir}/{f}"
        for f in os.listdir(preprocessed_dir)
        if f.endswith("epo.fif")
    ]

    # Limit to select subjects if necessary
    if subjects_lst is not None:
        epochs_fnames = [
            f
            for f in epochs_fnames
            if f.split("/")[-1].split("-")[0].split("_")[-1] in subjects_lst
        ]
    print(epochs_fnames)
    # Process each subject's data
    for epochs_fname in sorted(epochs_fnames):
        try:
            spec_decomp_and_param_one_subject(epochs_fname)
        except fooof.core.errors.DataError:
            print(f"Error processing Subject {epochs_fname}")
            continue


if __name__ == "__main__":
    # Process select task if given as command line argument
    subjects = None
    if len(sys.argv) > 1:
        subjects = sys.argv[1:]
    spec_decomp_and_param_all_subjects(subjects_lst=subjects)
