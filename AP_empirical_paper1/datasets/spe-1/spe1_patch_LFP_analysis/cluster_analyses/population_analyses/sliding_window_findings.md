# Sliding Window LFP Analysis — Key Findings

**Dataset**: spe-1 | **n = 43 cells** (7 priority, 36 non-priority)  
**Approach**: Time-resolved sliding window (Cohen's d per window, significant windows only)  
**LFP features used**: Aperiodic Exponent, Gamma AUC, Theta AUC, LFP Mean Amplitude, r²  
**Dropped (0% yield, confirmed redundant)**: Aperiodic Offset, LFP Spectral Exponent, LFP Std

---

## Main Finding: Near-Universal LFP Coupling Across Spike Waveform Clusters

Unlike the pre/post analysis, the sliding window finds **strong, consistent population-level effects** across nearly all spike features and LFP features. This is the primary analysis for characterising how spike waveform variability relates to network state.

---

## LFP Mean Amplitude is the Dominant Feature

The strongest and most universal result in the dataset:

| Subset | Spike feature | Yield |
|--------|--------------|-------|
| Priority | All spike features | **100%** |
| NP | log_ISI | 88% (23/26) |
| NP | peak_amp | 89% (8/9) |
| NP | spk_times_ms | 100% (10/10) |
| All combined | log_ISI | 91% (29/32) |
| All combined | spk_times_ms | **100%** (15/15) |

Essentially every cell that shows spike waveform clustering also shows significant LFP amplitude modulation in a time window around the spike. This is the cleanest population-level result in the dataset.

Effect sizes for LFP Mean Amplitude are also the largest of any LFP feature — individual cells reach Cohen's d up to 1.6 (log_ISI). Most other features sit in the 0.1–0.4 range.

---

## Spectral Features are Also Strongly Coupled

All four spectral features show high yield, confirming this is not just an amplitude effect:

| LFP Feature | All-cells yield range (across spike features) |
|-------------|----------------------------------------------|
| Aperiodic Exponent | 77–100% |
| Gamma AUC | 70–100% |
| Theta AUC | 60–92% |
| r-squared | 80–100% |

The near-identical yield across priority and NP subsets for all features confirms these are **genuine population-level effects**, not priority-cell artefacts.

---

## Feature Redundancy Confirmed

Aperiodic Offset, LFP Spectral Exponent, and LFP Std all show **0% yield everywhere** — across all spike features and both subsets. These features carry no unique information once the others are included. The decision to drop them was correct.

---

## Timing: Effects are Peri-Spike

The temporal density plots show significant windows concentrated around t = 0 (spike peak), with tails extending roughly ±0.25 s. Effects are not evenly distributed across the full window — they are tightest around the spike itself. This means the LFP-cluster coupling is primarily a **peri-spike phenomenon**, not a sustained background-state effect.

---

## Grand Average Traces: Consistent Directional Divergence

The grand average LFP Mean Amplitude traces show clean divergence between high and low cluster groups that is **consistent in direction** across cells:

- High ISI / high amplitude spikes → more negative peri-spike LFP amplitude dip
- Low ISI / low amplitude spikes → shallower LFP dip

The traces look similar in shape but offset in magnitude, meaning it's not that one cluster drives a different LFP waveform — both show the typical peri-spike dip, but its depth differs.

---

## How the Sliding Window Differs from Pre/Post

The pre/post analysis showed mostly null results for between-group effects. The sliding window finds the opposite — very strong effects. The key difference is that:

- Sliding window targets the **most significant time window** per (cell × feature) pair rather than averaging over fixed ±0.5 s windows
- LFP Mean Amplitude in the sliding window captures the sharp peri-spike LFP event, which is diluted when averaged over large pre/post windows
- The pre/post analysis is better suited to detecting **sustained state differences** (e.g., does one cluster occur in a persistently higher-exponent state?); the sliding window is better suited to detecting **event-locked coupling** around the spike

---

## Critical Qualification: Directionality is Heterogeneous

While yield is high, the **direction** of effects (which cluster has higher vs lower LFP) is not consistent across cells. This is visible in the grand average traces — when averaged across the full population, traces partially cancel. The effect is real and strong at the cell level, but the *sign* varies. This means:

- The LFP amplitude is coupled to spike cluster identity in most cells
- But "high cluster" = higher LFP in some cells and lower LFP in others
- Population-level conclusions about direction require stratification by cell type, recording modality, or other metadata (→ metadata notebook)

---

## Summary

| Finding | Strength |
|---------|----------|
| LFP amplitude coupled to spike clusters | **Very strong** — near-universal, large effect sizes |
| Spectral features (exponent, gamma, theta, r²) coupled | **Strong** — consistent 70–100% yield |
| Effects are peri-spike in timing | **Clear** — density peaks at t ≈ 0 |
| Effect direction consistent across population | **Weak** — heterogeneous, requires metadata stratification |
| Offset, LFP Std, LFP Spectral Exponent informative | **None** — 0% yield, definitively redundant |
