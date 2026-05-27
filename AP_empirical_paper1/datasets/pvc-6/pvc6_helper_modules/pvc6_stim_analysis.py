from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_val_predict, permutation_test_score
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
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


def random_forest_stim(X, y, X_train, X_test, y_train, y_test):
    """Random forest with grid search + bootstrapped accuracy."""
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
    accs = bootstrap_model(best_model, X_train, y_train, X_test, y_test)
    return best_model, accs


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

    # ── Bootstrap for coefficient CIs ────────────────────────────────────────
    bootstrapped_coefs       = []
    bootstrapped_r2          = []
    bootstrapped_adjusted_r2 = []

    for _ in range(bootstraps):
        X_res, y_res = resample(X, y, random_state=None)
        X_res_z = StandardScaler().fit_transform(X_res)
        model.fit(X_res_z, y_res)
        bootstrapped_coefs.append(model.coef_)
        r2         = model.score(X_res_z, y_res)
        adj_r2     = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
        bootstrapped_r2.append(r2)
        bootstrapped_adjusted_r2.append(adj_r2)

    bootstrapped_coefs       = np.array(bootstrapped_coefs)
    bootstrapped_r2          = np.array(bootstrapped_r2)
    bootstrapped_adjusted_r2 = np.array(bootstrapped_adjusted_r2)

    lower_bound     = np.percentile(bootstrapped_coefs, 2.5,  axis=0)
    upper_bound     = np.percentile(bootstrapped_coefs, 97.5, axis=0)
    standard_errors = np.std(bootstrapped_coefs, axis=0)
    p_values        = np.array([ttest_1samp(bootstrapped_coefs[:, i], 0)[1]
                                for i in range(bootstrapped_coefs.shape[1])])

    r2_mean          = np.mean(bootstrapped_r2)
    r2_ci            = np.percentile(bootstrapped_r2, [2.5, 97.5])
    adjusted_r2_mean = np.mean(bootstrapped_adjusted_r2)
    adjusted_r2_ci   = np.percentile(bootstrapped_adjusted_r2, [2.5, 97.5])

    return {
        # ── original keys (unchanged for backward compat) ──
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


# ── Stimulus onset analysis ───────────────────────────────────────────────────

def compute_stim_lag_correlations(f, fs, df_pink_raw, one_ms,
                                   window_ms=5, max_lag_ms=100,
                                   features=None):
    """
    Slide a pre-spike stimulus window from 0 to max_lag_ms before the spike
    inflection point and correlate mean stimulus amplitude with each waveform
    feature. Reveals the temporal window within which input drive shapes the AP.

    Parameters
    ----------
    f           : open h5py.File handle for the recording
    fs          : sampling rate (Hz)
    df_pink_raw : pink-noise rows from df — must retain 'sweep', 'spike_num',
                  'inflection_time', and all waveform feature columns
    one_ms      : samples per millisecond (= fs // 1000)
    window_ms   : width of each sliding window (ms)
    max_lag_ms  : how far back before inflection to search (ms)
    features    : waveform features to correlate against (default: 6 standard)

    Returns
    -------
    lag_centers : (n_lags,)            ms before inflection point (window centers)
    r_vals      : (n_lags, n_features) Pearson r
    p_vals      : (n_lags, n_features) p-values
    features    : list of feature names (same order as r_vals columns)
    """
    if features is None:
        features = ['ramp_amp', 'inflection_time', 'peak_amp',
                    'peak_sharpness', 'exp_lambda', 'exp_const']

    thresh_mv = -10
    thresh_ms = one_ms * 1

    lag_starts  = np.arange(0, max_lag_ms, window_ms)
    lag_centers = lag_starts + window_ms / 2.0
    n_lags      = len(lag_centers)
    n_spikes    = len(df_pink_raw)

    stim_at_lag  = np.full((n_spikes, n_lags), np.nan)
    idx_to_pos   = {idx: pos for pos, idx in enumerate(df_pink_raw.index)}

    for sweep_id, sweep_group in df_pink_raw.groupby('sweep'):
        dset = f['Sweep_' + str(int(sweep_id))]
        stim = np.array(dset[:, 0])
        data = np.array(dset[:, 1])

        idx_peaks, _ = find_peaks(data, height=thresh_mv, distance=thresh_ms)

        for orig_idx, row in sweep_group.iterrows():
            spike_num = int(row['spike_num'])
            if spike_num >= len(idx_peaks):
                continue

            peak_idx = idx_peaks[spike_num]
            # inflection_time is stored as ms from inflection to peak
            infl_idx = peak_idx - int(row['inflection_time'] * one_ms)
            pos      = idx_to_pos[orig_idx]

            for li, lag_start_ms in enumerate(lag_starts):
                end   = infl_idx - int(lag_start_ms * one_ms)
                start = end - int(window_ms * one_ms)
                if start < 0 or end > len(stim):
                    continue
                stim_at_lag[pos, li] = np.mean(stim[start:end])

    r_vals = np.full((n_lags, len(features)), np.nan)
    p_vals = np.full((n_lags, len(features)), np.nan)

    for li in range(n_lags):
        for fi, feat in enumerate(features):
            stim_lag  = stim_at_lag[:, li]
            feat_vals = df_pink_raw[feat].values.astype(float)
            valid     = np.isfinite(stim_lag) & np.isfinite(feat_vals)
            if valid.sum() > 10:
                r, p = pearsonr(stim_lag[valid], feat_vals[valid])
                r_vals[li, fi] = r
                p_vals[li, fi] = p

    return lag_centers, r_vals, p_vals, features


def f_test_r2(r2_small, r2_big, p_small, p_big, n):
    """F-test whether the larger model explains significantly more variance than the smaller one."""
    import scipy.stats as stats
    num   = (r2_big - r2_small) / (p_big - p_small)
    denom = (1 - r2_big) / (n - p_big - 1)
    F     = num / denom
    p     = 1 - stats.f.cdf(F, p_big - p_small, n - p_big - 1)
    return F, p
