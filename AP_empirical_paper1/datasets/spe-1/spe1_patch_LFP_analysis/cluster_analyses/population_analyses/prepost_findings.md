# Pre/Post Spike Specparam Analysis — Key Findings

**Dataset**: spe-1 | **n = 43 cells** (7 priority, 36 non-priority) | **Windows**: pre (−0.5 to −0.05 s), post (+0.05 to +1.0 s)

---

## Main Finding: log_ISI × Post-Spike Aperiodic Exponent

The most robust and replicable result across the dataset:

| Subset | Cells significant | Yield |
|--------|------------------|-------|
| Priority | 3/7 | 43% |
| Non-priority | 12/29 | 41% |
| All combined | 15/36 | **42%** |

Spikes from different ISI clusters are followed by measurably different **aperiodic slopes (exponent)** in the ~1 second after the spike. This holds at ~40% yield regardless of whether the cell was selected as priority or not — the consistency across independent subsets is what makes this credible.

Secondary: log_ISI also shows elevated yield for **r²** (31% all cells) and **gamma AUC** (22%) in the post window.

---

## Pre-Window: Weak Across the Board

No spike feature shows consistent pre-spike LFP state differences between clusters. Effects are small (|g| < 0.2), highly variable, and centered at zero. The LFP state does not reliably differ before the spike depending on which cluster it belongs to.

---

## Within-Group: General Post-Spike Spectral Steepening

The aperiodic exponent **increases** after spikes for both cluster groups (positive d_z, visible across most spike features). Gamma power **decreases** post-spike, especially for peak_amp clusters. This is a general post-spike LFP change — not cluster-specific — and likely reflects the known post-spike hyperpolarization/state reset.

---

## No Interaction Effect

The A−B cluster difference does not consistently change from pre to post spike (~14–29% yield). Both clusters modulate the LFP similarly; they don't diverge differently around the spike.

---

## Waveform Cluster Difference Does Not Predict LFP Effect Strength

No significant Spearman ρ between nRMSE/cos_sim and pre/post effect sizes (all blue, BH-FDR corrected). Cells with more distinct waveform clusters do not show stronger LFP modulation. The LFP coupling is independent of how separable the waveforms are.

---

## Critical Qualification: ~40% ≠ Universal

The log_ISI finding is real but **heterogeneous** — it appears in ~40% of cells and is absent in ~60%. This is not a whole-population effect. The natural next question is: **what distinguishes cells that show this from those that don't?** → See metadata analysis.

---

## What This Means Biologically

- Spike waveform variability (amplitude, width, sharpness) is largely **not** tied to pre- or post-spike LFP network states at the population level
- **ISI history is the exception**: log_ISI clusters predict post-spike spectral state in ~40% of cells, which makes sense since ISI captures firing rate history — a quantity directly coupled to network dynamics
- The effect emerging *post-spike* (not pre) suggests this is not simply that different-state spikes look different; it may reflect the network consequences of different firing patterns
- The subset nature (~40%) calls for a metadata-driven follow-up to identify what predicts which cells show this coupling
