# Ridge Regression Analysis — Key Findings

**Dataset**: spe-1 | **n = 37 cells** (cells with ≥ 1 fitted ridge result)  
**Approach**: Per-cell 5-fold CV ridge regression; spike waveform features → LFP state features  
**Time windows**: Pre spike −55 → −5 ms; Post spike +5 → +55 ms; Baseline −200 → −100 ms  
**LFP targets**: 5 feature types × 5 window variants (abs, BL-corrected, Δ) = 25 targets  
**Predictor sets**: Waveform only, Log ISI only, Waveform + Log ISI  
**Population test**: One-sided Wilcoxon signed-rank on CV R² across cells (H₀: median R² = 0), BH-FDR q < 0.05

---

## Main Finding: Pre- and Post-Spike LFP Amplitude and Std Are Predictable from Waveform Features

Four LFP targets reach population significance under the Waveform-only predictor set:

| Target | Window | Wilcoxon p (FDR-corrected) |
|--------|--------|---------------------------|
| LFP Amplitude | Pre (−55 → −5 ms) | q < 0.05 |
| LFP Std | Pre (−55 → −5 ms) | q < 0.05 |
| LFP Amplitude | Post (+5 → +55 ms) | q < 0.05 |
| LFP Std | Post (+5 → +55 ms) | q < 0.05 |

Spectral features (aperiodic exponent, gamma AUC, theta AUC, r²) and baseline-corrected variants do not reach significance at the population level. This means the waveform-to-LFP relationship is strongest and most consistent for broadband amplitude measures directly around the spike, not for sustained spectral state.

The absolute (non-baseline-corrected) pre and post windows both survive — the signal is not an artefact of the specific baseline window choice.

---

## Effect Sizes Are Modest but Consistent

CV R² values are low in absolute terms (median R² typically 0.02–0.08 for significant targets), which is expected when predicting a continuous LFP feature from 8 spike waveform parameters in naturalistic recordings. The population significance reflects **cross-cell consistency** (median CV R² reliably above zero) rather than large single-cell effects.

---

## Beta Consistency: No Targets Survive FDR After Correction for All Predictors × Targets

One-sample Wilcoxon on beta weights (do the betas have a consistent non-zero sign across cells?) yields no FDR-significant (target, predictor) pairs at q < 0.05 after correcting across all 200+ comparisons. This is consistent with the directional heterogeneity observed in the sliding-window analysis — LFP amplitude is coupled to spike cluster identity in most cells, but the *direction* varies. See `metadata_findings.md` for the same pattern.

---

## Fraction Significant: Above-Chance Binomial Hits for Amplitude Targets

The binomial test (is the fraction of FDR-significant single-cell models greater than the expected false-positive rate?) shows above-chance yield for the significant amplitude and std targets, confirming the population Wilcoxon result is not driven by a small number of outlier cells.

---

## Log ISI Does Not Consistently Improve Prediction Beyond Waveform Features

The Waveform + Log ISI predictor set does not systematically produce more significant targets or higher mean R² than Waveform only across cells. Log ISI alone (as a single predictor) also does not reliably predict LFP amplitude. This contrasts with the sliding-window / pre-post analyses, where log_ISI clusters showed the strongest LFP coupling — suggesting that log_ISI's coupling to LFP state is captured at the cluster level (relative to within-cell high vs low groups) but not by the raw value of log ISI as a regression predictor.

---

## Cell Identity × R²: No Strong Predictors

No cell-level metadata variable (cell type PC/IN, recording type Juxta/WC, dark-neuron status, cortical depth, EAP visibility) significantly predicts mean CV R² after BH-FDR correction. This is consistent with the metadata analysis of the sliding-window pipeline, where recording methodology does not explain LFP coupling strength.

Sample size (n spikes per cell) is not a significant confound (Spearman ρ between n spikes and mean CV R² is near zero), ruling out the concern that cells with longer recordings drive the significant results.

---

## Regularisation: High Alpha (Strong Regularisation) Is the Norm

Best ridge alpha selected by cross-validation tends toward large values for most targets, indicating the regression needed substantial regularisation. This is consistent with the data being high-dimensional (8 spike features) relative to the effect size (modest waveform-to-LFP coupling). The pre/post LFP amplitude targets that survive significance select lower alpha on average, consistent with these being the targets where a genuine signal is present.

---

## Relationship to Sliding-Window and Pre/Post Results

| Analysis | LFP target | Finding |
|---|---|---|
| Sliding window (cluster comparison) | LFP Mean Amplitude | Near-universal coupling (70–100% yield) |
| Pre/post (cluster comparison) | LFP Exponent (post) | ~40% yield for log_ISI clusters |
| Ridge regression (continuous prediction) | LFP Amplitude + Std (pre & post) | 4 targets, modest but consistent R² |

The three analyses converge on **LFP amplitude** as the most tightly coupled LFP feature. The ridge regression is more conservative because it predicts the continuous LFP value from continuous waveform features — harder than testing whether cluster *groups* differ. The convergence across methods strengthens the conclusion.

---

## Summary

| Finding | Strength |
|---------|----------|
| Pre/post LFP amplitude predictable from waveform features | **Moderate** — 4 targets significant, modest R² |
| Spectral features (exponent, gamma, theta) predictable | **Weak** — no FDR-significant targets |
| Log ISI improves LFP prediction beyond waveform features | **None** — no systematic improvement |
| Beta weights consistent in direction across cells | **None** — heterogeneous, consistent with sliding-window result |
| Cell metadata predicts CV R² | **None** — all ns after BH-FDR |
| n spikes (sample size) is a confound | **None** — ρ ≈ 0 |
