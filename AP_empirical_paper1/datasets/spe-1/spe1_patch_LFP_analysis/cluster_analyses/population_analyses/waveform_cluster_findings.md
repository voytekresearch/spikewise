# Waveform Cluster Analysis — Key Findings

**Dataset**: spe-1 | **n = 43 cells** | **Spike features**: 11 | **Cluster metrics**: nRMSE, cos_sim, num_clusters, temporal_rho

---

## Main Finding: log_ISI is the Most Prevalent Clustering Feature, but Not the Most Distinctive

log_ISI shows clustering in **33–38% of cells** across all metadata groups — roughly double or triple the prevalence of any other single spike feature. Dark neurons push this even higher at **62.5%** prevalence. No other feature comes close in terms of how consistently it clusters across the population.

But prevalence and waveform distinctiveness are not the same thing. log_ISI clusters occupy the **low-nRMSE / high-cos-sim** region of the waveform variance landscape — spikes split into ISI-history groups are not morphologically different from each other. The two clusters look nearly identical in shape. The *grouping* is real, the *waveform difference* is minimal.

The features with the **largest waveform differences** are peak_width, peak_sharpness, inflection_time, and spk_times — these sit at high nRMSE and low cos_sim. They're less prevalent but when they cluster, the two waveform groups are genuinely distinct in shape.

---

## Clustering Prevalence by Spike Feature

| Spike feature | Prevalence range (across metadata groups) | nRMSE (mean) | cos_sim (mean) |
|--------------|------------------------------------------|--------------|----------------|
| log_ISI | 29–62% | ~0.04 | ~0.97 |
| peak_amp | 9–18% | ~0.09 | ~0.95 |
| peak_width | 8–28% | ~0.08 | ~0.92 |
| spk_times_ms | 12–13% | ~0.07 | ~0.95 |
| peak_sharpness | 6–13% | ~0.07 | ~0.93 |
| exp_lambda | 0–13% | ~0.07 | ~0.92 |
| inflection_time | 2–11% | ~0.06 | ~0.96 |
| inflection_amp | 0–8% | ~0.10 | ~0.95 |
| spk_times_idx | 4–10% | ~0.07 | ~0.92 |
| exp_const | 0–1% | — | — |

log_ISI prevalence is consistent across cell type, recording type, and clamp mode — it's not a subset artefact.

---

## Top Cells: Priority Selection Was Correct

The five highest mean-nRMSE cells are all priority cells: **c42 > c26 > c21 > c24 > c3** (mean nRMSE 0.13–0.18). The next two priority cells (c27, c4) also appear in the top 10. The priority selection process picked the cells with the most biologically meaningful waveform variability.

The waveform variance landscape shows clear separation: priority cells cluster in the high-nRMSE / low-cos-sim corner, while most non-priority cells cluster near (0.00, 0.99) — negligible amplitude and shape difference between clusters.

Top cell-feature pairs by combined nRMSE + cos_sim difference: c24 × log_ISI, c21 × peak_width, c21 × peak_amp, c27 × peak_sharpness, c42 × peak_width, c27 × inflection_amp.

---

## Metadata Does Not Predict Cluster Quality — Except EAP Visibility

After BH-FDR correction, **only one** metadata × cluster quality association survives:

**clear_EAP_waveform × cos_sim** (Spearman ρ = −0.381, \*)

Cells with a visible extracellular AP show **lower cosine similarity** between cluster waveforms — their clusters are more morphologically distinct. Everything else (cell type, patch type, recording type, clamp mode, dark neuron, cortical depth) is non-significant. This is actually reassuring: clustering quality is not systematically driven by how you recorded the cell. EAP visibility being the exception makes sense biologically — cells with clear APs likely have better signal isolation and more genuine waveform variability.

---

## Temporal Drift: Near-Universal, Feature-Specific Direction

**96/110 cell-feature pairs (87%) show significant temporal drift** (Spearman ρ ≠ 0, p < 0.05). Cluster membership shifts monotonically over recording time in the vast majority of cases. This is not a red flag per se — it reflects that many spike features change over the recording — but the *direction* is strongly feature-dependent:

- **peak_amp, peak_sharpness**: ρ ≈ −0.5 (getting smaller over time — consistent with electrode seal degradation or AHP accumulation)
- **peak_width, inflection_time, spk_times_ms, spk_times_idx**: ρ ≈ +0.75 (increasing over time)
- **log_ISI**: ρ ≈ 0, centered at zero (no drift)

This feature-specific directionality is striking. The Kruskal-Wallis test across features is highly significant (p < 0.0001). log_ISI is the only feature that shows essentially no temporal structure — its cluster identity is not a recording-time artefact.

**Only cortical depth predicts drift magnitude** (Spearman ρ = 0.17, p = 0.036). Deeper recordings drift slightly more, possibly reflecting greater mechanical instability. No other metadata variable predicts drift (all p_fdr > 0.1).

---

## Temporal Drift Does Not Inflate Waveform Differences

A key concern: do cells with more drift show bigger cluster differences, meaning the "cluster separation" is partly just early-vs-late recording? The answer is **no**:

- temporal_rho × nRMSE: Spearman ρ = −0.02, ns
- temporal_rho × cos_sim: Spearman ρ = +0.02, ns

Cells with significant drift and cells without drift show statistically identical nRMSE and cos_sim distributions (p_fdr > 0.7). The waveform cluster differences are real features of the data, not temporal recording artefacts.

---

## Population Waveform Grid

The grid shows the full picture across all 43 cells. Priority cells (orange border) consistently show visible separation between the two cluster waveforms. For most non-priority cells the two traces overlap almost completely — the clusters exist statistically but the waveform difference is negligible. This validates the priority cell framework: the ~7 priority cells are where genuine morphological variability is concentrated.

---

## I. Within-cell vs Between-cell Waveform Distances

A key context question: how does within-cell cluster variability compare to the variability between completely different neurons?

| | nRMSE (median) | nRMSE IQR | cos_sim (median) |
|---|---|---|---|
| Within-cell clusters | 0.049 | [0.022, 0.091] | 0.980 |
| Between-cell pairs | 0.264 | [0.189, 0.318] | 0.763 |

**Within-cell cluster differences are ~5× smaller than between-cell differences in amplitude (nRMSE) and sit at much higher cosine similarity.** On average, the clusters within a neuron look far more like each other than two different neurons do.

This is an important calibration: the waveform clusters are capturing **sub-neuronal variability**, not cell-identity-scale differences. The clusters are statistically real and functionally meaningful (as the LFP results show), but their morphological footprint is small relative to what separates distinct neurons.

The exception is the top priority cells. The highest within-cell nRMSE values (c42, c27, c21: nRMSE 0.13–0.20) overlap with the **lower tail of the between-cell distribution** (25th percentile ≈ 0.189). For these cells, within-cell cluster separation reaches the floor of between-cell distinguishability — the waveform variability within those neurons is comparable to the difference between some pairs of distinct neurons.

---

## Summary

| Finding | Strength |
|---------|----------|
| log_ISI is the most prevalent clustering feature | **Strong** — 33–38% across groups, 62.5% for dark neurons |
| log_ISI clusters are morphologically similar (low nRMSE) | **Clear** — high prevalence, low waveform difference |
| Peak-width / sharpness / inflection-time have largest waveform separation | **Clear** — high nRMSE, low cos_sim, lower prevalence |
| Priority cell selection captured the highest-variability cells | **Confirmed** — top 5–7 cells by mean nRMSE are all priority cells |
| Metadata predicts cluster quality | **Weak** — only EAP visibility × cos_sim survives BH-FDR |
| Temporal drift is near-universal | **Strong** — 87% of cell-feature pairs |
| Drift direction is feature-specific (amplitude decreases, timing increases) | **Very strong** — Kruskal-Wallis p < 0.0001 |
| Temporal drift inflates waveform differences | **None** — ρ ≈ 0, ns |
| Within-cell cluster differences comparable to between-cell distances | **No** — ~5× smaller on average; only top priority cells approach the between-cell floor |
