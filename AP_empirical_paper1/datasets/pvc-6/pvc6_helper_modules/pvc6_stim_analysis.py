from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_val_predict, permutation_test_score
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_1samp, pearsonr
from scipy.signal import find_peaks
from statsmodels.stats.multitest import fdrcorrection
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from pvc6_plotting import *


# ── Categorical stimulus classifiers ─────────────────────────────────────────

def bootstrap_model(model, X_train, y_train, X_test, y_test, n_bootstraps=1000):
    """Bootstrap accuracy distribution for a fitted classifier."""
    bootstrapped_accuracies = []
    for _ in range(n_bootstraps):
        X_resampled, y_resampled = resample(X_train, y_train, random_state=None)
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)
        bootstrapped_accuracies.append(accuracy_score(y_test, y_pred))
    return np.array(bootstrapped_accuracies)


def logistic_regression_stim(X, y, X_train, X_test, y_train, y_test):
    """Multinomial logistic regression + bootstrapped accuracy."""
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    accs = bootstrap_model(model, X_train, y_train, X_test, y_test)
    return model, accs


def svm_stim(X, y, X_train, X_test, y_train, y_test):
    """Linear SVM with C grid search + bootstrapped accuracy."""
    param_grid = {'C': [0.1, 1, 10, 100]}
    grid_search = GridSearchCV(SVC(kernel='linear', probability=True), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    accs = bootstrap_model(best_model, X_train, y_train, X_test, y_test)
    return best_model, accs


def random_forest_stim(X, y, X_train, X_test, y_train, y_test, n_bootstraps=1000):
    """Random forest with grid search + bootstrapped accuracy and feature importances."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    bootstrapped_accuracies = []
    bootstrapped_importances = []
    for _ in range(n_bootstraps):
        X_resampled, y_resampled = resample(X_train, y_train, random_state=None)
        best_model.fit(X_resampled, y_resampled)
        y_pred = best_model.predict(X_test)
        bootstrapped_accuracies.append(accuracy_score(y_test, y_pred))
        bootstrapped_importances.append(best_model.feature_importances_)

    return best_model, np.array(bootstrapped_accuracies), np.array(bootstrapped_importances)


# ── Continuous stimulus regression ───────────────────────────────────────────

def run_ridge_regression_kfold(X, y, n_splits=5, random_state=42, bootstraps=1000,
                                n_perm=1000, alphas=None):
    """
    K-fold Ridge regression — aligned with the spe-1 pipeline for paper consistency.

    Changes vs original:
      - Features Z-scored (StandardScaler) so beta weights are in comparable units
      - Alpha tuned via RidgeCV instead of fixed α=1
      - Permutation test (n_perm) for model-level significance
      - Bootstrapped 95% CIs on coefficients and R² retained

    Returns dict — all original keys preserved; new keys added:
      best_alpha, p_val_perm, null_mean, null_std
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 100)

    kf        = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # ── Tune alpha on Z-scored features ──────────────────────────────────────
    X_z        = StandardScaler().fit_transform(X)
    alpha_cv   = RidgeCV(alphas=alphas, fit_intercept=True)
    alpha_cv.fit(X_z, y)
    best_alpha = float(alpha_cv.alpha_)

    # ── Pipeline (scaler inside CV folds to prevent leakage) ─────────────────
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha, fit_intercept=True))

    y_pred_cv          = cross_val_predict(pipe, X, y, cv=kf)
    scores             = cross_val_score(pipe, X, y, cv=kf, scoring='r2')
    adjusted_r2_scores = 1 - ((1 - scores) * (n_samples - 1) / (n_samples - n_features - 1))

    print(f"Cross-validated R-squared scores: {scores}")
    print(f"Average R-squared: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Adjusted R-squared scores: {adjusted_r2_scores}")
    print(f"Average Adjusted R-squared: {adjusted_r2_scores.mean():.3f} ± {adjusted_r2_scores.std():.3f}")
    print(f"Best alpha: {best_alpha:.4g}")

    # ── Permutation test for model significance ───────────────────────────────
    cv_score, perm_scores, p_val_perm = permutation_test_score(
        pipe, X, y, cv=kf, n_permutations=n_perm,
        scoring='r2', random_state=random_state, n_jobs=1,
    )
    print(f"Permutation p-value: {p_val_perm:.4f}")

    # ── Full-data fit on Z-scored X for coefficient extraction ────────────────
    model = Ridge(alpha=best_alpha, fit_intercept=True)
    model.fit(X_z, y)
    coefficients  = model.coef_
    feature_names = X.columns

    # ── Bootstrap coefficient CIs (training bootstrap — standard for coef SE) ──
    rng = np.random.default_rng(random_state)
    bootstrapped_coefs       = []
    all_indices = np.arange(n_samples)

    for _ in range(bootstraps):
        boot_idx = rng.choice(all_indices, size=n_samples, replace=True)
        X_boot   = X.iloc[boot_idx]; y_boot = y.iloc[boot_idx]
        scaler_b = StandardScaler()
        model.fit(scaler_b.fit_transform(X_boot), y_boot)
        bootstrapped_coefs.append(model.coef_)

    bootstrapped_coefs = np.array(bootstrapped_coefs)
    lower_bound     = np.percentile(bootstrapped_coefs, 2.5,  axis=0)
    upper_bound     = np.percentile(bootstrapped_coefs, 97.5, axis=0)
    standard_errors = np.std(bootstrapped_coefs, axis=0)
    p_values        = np.array([ttest_1samp(bootstrapped_coefs[:, i], 0)[1]
                                for i in range(bootstrapped_coefs.shape[1])])

    # ── Bootstrap R² from CV predictions (unbiased, stable CIs) ─────────────
    # Resample (y, y_pred_cv) pairs — no re-fitting, CIs on held-out R².
    y_arr      = np.asarray(y)
    yhat_arr   = np.asarray(y_pred_cv)
    boot_r2    = []
    boot_adjr2 = []
    for _ in range(bootstraps):
        idx   = rng.choice(n_samples, size=n_samples, replace=True)
        r2    = r2_score(y_arr[idx], yhat_arr[idx])
        adj   = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
        boot_r2.append(r2)
        boot_adjr2.append(adj)

    bootstrapped_r2          = np.array(boot_r2)
    bootstrapped_adjusted_r2 = np.array(boot_adjr2)
    r2_mean          = float(np.mean(bootstrapped_r2))
    r2_ci            = np.percentile(bootstrapped_r2, [2.5, 97.5])
    adjusted_r2_mean = float(np.mean(bootstrapped_adjusted_r2))
    adjusted_r2_ci   = np.percentile(bootstrapped_adjusted_r2, [2.5, 97.5])

    return {
        # ── original keys (unchanged for backward compat) ──
        "y_true":                     np.asarray(y),
        "y_pred_cv":                  y_pred_cv,
        "coefficients":               coefficients,
        "feature_names":              feature_names,
        "r2_scores":                  scores,
        "adjusted_r2_scores":         adjusted_r2_scores,
        "bootstrapped_coefs":         bootstrapped_coefs,
        "bootstrapped_r2":            bootstrapped_r2,
        "bootstrapped_adjusted_r2":   bootstrapped_adjusted_r2,
        "ci_lower":                   lower_bound,
        "ci_upper":                   upper_bound,
        "standard_errors":            standard_errors,
        "p_values":                   p_values,
        "r2_mean":                    r2_mean,
        "r2_ci":                      r2_ci,
        "adjusted_r2_mean":           adjusted_r2_mean,
        "adjusted_r2_ci":             adjusted_r2_ci,
        # ── new keys aligned with spe-1 ──
        "best_alpha":                 best_alpha,
        "p_val_perm":                 float(p_val_perm),
        "null_mean":                  float(np.mean(perm_scores)),
        "null_std":                   float(np.std(perm_scores)),
    }


def apply_fdr_pvc6(results_dict, q=0.05):
    """
    Benjamini-Hochberg FDR correction across a set of ridge regression results.
    Mirrors spe-1's apply_fdr().

    Parameters
    ----------
    results_dict : dict
        name → ridge_results (output of run_ridge_regression_kfold).
        E.g. {'stim_exp': r1, 'stim_mean': r2, 'stim_std': r3}
    q : float
        FDR threshold (default 0.05)

    Returns
    -------
    results_dict with 'p_val_fdr' and 'sig_fdr' added in-place.
    """
    keys      = list(results_dict.keys())
    raw_pvals = np.array([results_dict[k].get('p_val_perm', np.nan) for k in keys])
    pvals_in  = np.where(np.isfinite(raw_pvals), raw_pvals, 1.0)
    rejected, pvals_fdr = fdrcorrection(pvals_in, alpha=q, method='indep')

    for k, pfdr, rej in zip(keys, pvals_fdr, rejected):
        results_dict[k]['p_val_fdr'] = float(pfdr)
        results_dict[k]['sig_fdr']   = bool(rej)

    n_raw = int(np.sum(raw_pvals < 0.05))
    n_fdr = int(np.sum(rejected))
    print(f"FDR (BH, q={q}): {n_raw}/{len(keys)} raw p<0.05 → {n_fdr}/{len(keys)} after correction")
    for k in keys:
        r2   = results_dict[k].get('r2_mean', np.nan)
        p    = results_dict[k].get('p_val_perm', np.nan)
        pfdr = results_dict[k].get('p_val_fdr', np.nan)
        sig  = '*' if results_dict[k].get('sig_fdr') else ' '
        print(f"  {k:<15}  R²={r2:.3f}  p_perm={p:.3f}  p_fdr={pfdr:.3f} {sig}")

    return results_dict


def recompute_stim_features(f, fs, df_pink_raw, one_ms, window_ms=50, offset_ms=0):
    """
    Re-extract pre-inflection stimulus statistics for each pink-noise spike
    using a configurable window width and optional offset.

    The window covers [infl_idx - (offset_ms + window_ms) : infl_idx - offset_ms].
    offset_ms=0 (default) gives the standard window ending at the inflection.
    Set offset_ms=500, window_ms=100 for a pre-stimulus control window (500–600ms
    before the spike, outside the 500ms stimulus epoch).

    Parameters
    ----------
    f            : open h5py.File handle for the recording
    fs           : sampling rate (Hz)
    df_pink_raw  : pink-noise rows from df — must retain 'sweep', 'spike_num',
                   'inflection_time', and all waveform feature columns
    one_ms       : samples per millisecond (= fs // 1000)
    window_ms    : window width in ms
    offset_ms    : shift the window this many ms further back from the inflection

    Returns
    -------
    pd.DataFrame indexed identically to df_pink_raw with columns:
        stim_mean_<suffix>, stim_std_<suffix>, stim_exp_<suffix>
    """
    import pandas as _pd
    from scipy.signal import find_peaks as _find_peaks
    from neurodsp import spectral as _spectral

    thresh_mv = -10
    thresh_ms = one_ms * 1

    n = len(df_pink_raw)
    mean_vals = np.full(n, np.nan)
    std_vals  = np.full(n, np.nan)
    exp_vals  = np.full(n, np.nan)

    idx_to_pos = {idx: pos for pos, idx in enumerate(df_pink_raw.index)}

    for sweep_id, sweep_group in df_pink_raw.groupby('sweep'):
        dset = f['Sweep_' + str(int(sweep_id))]
        stim = np.array(dset[:, 0])
        data = np.array(dset[:, 1])

        idx_peaks, _ = _find_peaks(data, height=thresh_mv, distance=thresh_ms)

        for orig_idx, row in sweep_group.iterrows():
            spike_num = int(row['spike_num'])
            if spike_num >= len(idx_peaks):
                continue

            peak_idx = idx_peaks[spike_num]
            infl_idx = peak_idx - int(row['inflection_time'] * one_ms)
            pos      = idx_to_pos[orig_idx]

            end   = infl_idx - int(offset_ms * one_ms)
            start = end - int(window_ms * one_ms)
            if start < 0 or end > len(stim):
                continue

            w_stim = stim[start:end]
            mean_vals[pos] = np.mean(w_stim)
            std_vals[pos]  = np.std(w_stim)

            try:
                fxx, pxx = _spectral.compute_spectrum(
                    w_stim, fs, method='welch',
                    window='hann', nperseg=len(w_stim),
                )
                fxx, pxx = fxx[1:], pxx[1:]
                if len(fxx) > 2:
                    slope, _ = np.polyfit(np.log10(fxx), np.log10(pxx), 1)
                    exp_vals[pos] = -slope
            except Exception:
                pass

    suffix = f'{window_ms}ms' if offset_ms == 0 else f'ctrl_{offset_ms}to{offset_ms + window_ms}ms'
    return _pd.DataFrame(
        {f'stim_mean_{suffix}': mean_vals,
         f'stim_std_{suffix}':  std_vals,
         f'stim_exp_{suffix}':  exp_vals},
        index=df_pink_raw.index,
    )


def f_test_r2(r2_small, r2_big, p_small, p_big, n):
    """F-test whether the larger model explains significantly more variance than the smaller one."""
    import scipy.stats as stats
    num   = (r2_big - r2_small) / (p_big - p_small)
    denom = (1 - r2_big) / (n - p_big - 1)
    F     = num / denom
    p     = 1 - stats.f.cdf(F, p_big - p_small, n - p_big - 1)
    return F, p


_DROP_COLS = ['stim_exp', 'stim_mean', 'stim_std', 'log_isi']


def prepare_window_df(df_pink_filtered, df_pink_raw_filtered, f, fs, one_ms, window_ms,
                      pickle_dir=None, force_recompute=False):
    """Return a copy of df_pink_filtered with stim features recomputed for window_ms.

    For the 5 ms default window, returns df_pink_filtered unchanged.
    Results are cached per window_ms in pickle_dir to avoid recomputing on repeat calls.
    """
    if window_ms == 5:
        return df_pink_filtered.copy()

    import pickle as _pkl
    from pathlib import Path as _Path

    _cache = None
    if pickle_dir is not None:
        _cache_path = _Path(pickle_dir) / f'_stim_extra_{window_ms}ms.pkl'
        if not force_recompute and _cache_path.exists():
            print(f"  Loading stim features ({window_ms}ms) from cache...")
            with open(_cache_path, 'rb') as _fh:
                stim_extra = _pkl.load(_fh)
        else:
            print(f"  Computing stim features ({window_ms}ms)...")
            stim_extra = recompute_stim_features(f, fs, df_pink_raw_filtered, one_ms, window_ms=window_ms)
            with open(_cache_path, 'wb') as _fh:
                _pkl.dump(stim_extra, _fh)
            print(f"  Cached → {_cache_path.name}")
    else:
        print(f"  Computing stim features ({window_ms}ms) — pass pickle_dir to cache...")
        stim_extra = recompute_stim_features(f, fs, df_pink_raw_filtered, one_ms, window_ms=window_ms)

    df_w = df_pink_filtered.copy()
    df_w['stim_mean'] = stim_extra[f'stim_mean_{window_ms}ms']
    df_w['stim_std']  = stim_extra[f'stim_std_{window_ms}ms']
    df_w['stim_exp']  = stim_extra[f'stim_exp_{window_ms}ms']
    return df_w.dropna(subset=['stim_mean'])


def run_window_expansion(windows_ms, df_pink_filtered, df_pink_raw_filtered,
                         f, fs, one_ms, existing=None, rng=None):
    """Compute ridge regression (real + shuffle) for each window in windows_ms.

    Only missing keys are computed; already-present entries in *existing* are kept.
    Returns the updated results dict.
    """
    results = dict(existing or {})
    if rng is None:
        rng = np.random.default_rng(42)
    targets = ['stim_mean', 'stim_std', 'stim_exp']

    missing_windows = [w for w in windows_ms
                       if any(f'{w}ms_{t}' not in results for t in targets)]
    missing_shuffle = [w for w in windows_ms
                       if any(f'shuf_{w}ms_{t}' not in results for t in targets)]

    if missing_windows:
        print(f'Computing missing windows: {missing_windows}')
        for wms in missing_windows:
            df_w = prepare_window_df(df_pink_filtered, df_pink_raw_filtered, f, fs, one_ms, wms)
            for target in targets:
                data_t = df_w.dropna(subset=[target])
                drop   = [c for c in _DROP_COLS if c in data_t.columns]
                X_t    = data_t.drop(columns=drop)
                y_t    = data_t[target]
                print(f'\n--- {wms} ms | {target} ---')
                results[f'{wms}ms_{target}'] = run_ridge_regression_kfold(X_t, y_t)

    if missing_shuffle:
        print(f'\nComputing shuffle controls for windows: {missing_shuffle}')
        for wms in missing_shuffle:
            df_w = prepare_window_df(df_pink_filtered, df_pink_raw_filtered, f, fs, one_ms, wms)
            for target in targets:
                data_t = df_w.dropna(subset=[target])
                drop   = [c for c in _DROP_COLS if c in data_t.columns]
                X_t    = data_t.drop(columns=drop)
                y_shuf = pd.Series(rng.permutation(data_t[target].values), index=data_t.index)
                print(f'\n--- shuf {wms} ms | {target} ---')
                results[f'shuf_{wms}ms_{target}'] = run_ridge_regression_kfold(X_t, y_shuf)

    if not missing_windows and not missing_shuffle:
        print('All windows (including shuffle controls) already computed.')

    return results
