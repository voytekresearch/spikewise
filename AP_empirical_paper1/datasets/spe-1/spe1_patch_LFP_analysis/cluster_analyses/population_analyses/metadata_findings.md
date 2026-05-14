# Metadata Analysis — Key Findings

**Dataset**: spe-1 | **n = 43 cells** (7 priority, 36 non-priority)  
**Metadata variables**: cell_type (PC/IN), recording_type (Juxta/WC), clamp_mode (IC/VC), dark_neuron (yes/no), EAP visibility (yes/no), cortical depth (continuous)  
**LFP features**: Aperiodic Exponent, r², Theta AUC, Gamma AUC, LFP Mean Amplitude

---

## Main Finding: Cell Type (PC vs IN) Predicts LFP Coupling Strength

The only metadata variable that consistently and significantly predicts LFP-spike coupling strength is **cell type**:

**PCs show significantly larger |Cohen's d| than INs** for all LFP features except Theta AUC (BH-FDR corrected):

| LFP Feature | p_fdr (all cells) |
|-------------|-------------------|
| Aperiodic Exponent | 0.003 ** |
| r² | 0.006 ** |
| Gamma AUC | < 0.001 *** |
| LFP Mean Amplitude | 0.002 ** |
| Theta AUC | 0.121–0.171 ns |

This replicates across subsets (NP and all-cells) and is the strongest result in this analysis. INs in this dataset show systematically weaker LFP-spike coupling for broadband and gamma spectral features than PCs.

Theta AUC is the exception — the PC/IN difference does not reach significance for theta, suggesting theta-range modulation may be less cell-type specific in this dataset.

---

## Everything Else Is Not Significant

All other metadata variables fail to predict LFP coupling strength after BH-FDR correction in the all-cells subset:

| Metadata | All LFP features | Note |
|----------|-----------------|------|
| recording_type (Juxta/WC) | all ns (p_fdr > 0.20) | Juxta numerically higher, but high variance |
| clamp_mode (IC/VC) | all ns (p_fdr > 0.55) | IC numerically larger for most features |
| dark_neuron | all ns (p_fdr > 0.70) | No systematic difference |
| EAP visibility | all ns (p_fdr > 0.17) | Counterintuitively, not-visible cells have higher median |
| cortical depth | all ns | ρ estimates near 0, wide CIs — no depth gradient |

This is an important negative result: the LFP coupling strength is not explained by how the cell was recorded (patch type, clamp, recording modality) or where it sits in the cortical column. The cell_type finding is not a confound of recording differences.

---

## 3-Way Analysis (Metadata × LFP × Spike Feature): Nothing Survives

The 3-way Kruskal-Wallis tests — asking whether metadata predicts which specific LFP feature × spike feature combination shows the strongest effect — find **nothing significant** after BH-FDR correction in any subset. All heatmap cells show p_fdr ≈ 1.

This means:
- While PCs show stronger LFP coupling overall, this advantage is uniformly distributed across spike features and LFP features — not specific to any particular LFP-spike pairing
- The question "does cell type predict whether log_ISI vs peak_amp drives the stronger LFP effect?" has no significant answer in this dataset

This is likely an underpowered question given the combination of small cell counts and the already-high multiplicity burden.

---

## Effect Direction: Mostly Consistent, Not Explained by Metadata

Across all directionality plots, signed Cohen's d values are **predominantly positive** (high-cluster group shows higher LFP feature values). The directional heterogeneity flagged in the sliding window analysis is not well explained by any single metadata variable:

- **LFP Mean Amplitude × log_ISI**: all metadata groups sit above zero — the "high ISI → higher LFP amplitude" direction holds regardless of cell type, recording type, or dark neuron status
- **Aperiodic Exponent × log_ISI**: same pattern — both IN and PC are positive; PCs show larger effects (~0.22 median) than INs (~0.12), consistent with the 2-way finding
- The residual heterogeneity in the direction plots is within-group scatter, not between-group reversal

The conclusion from sliding window that direction is heterogeneous is **not explained by any measured metadata variable**.

---

## Pre/Post Exponent × Metadata: Nothing Predicts the 40% Yield

The pre/post analysis found log_ISI clusters predict post-spike aperiodic exponent in ~40% of cells. The metadata breakdown of the Hedges' g distribution for this effect finds:

| Metadata | p_fdr | Pattern |
|----------|-------|---------|
| cell_type | ns | Both IN and PC centered at g ≈ 0; wide scatter for both |
| recording_type | ns | Juxta highly variable around 0; WC tightly clustered at 0 |
| clamp_mode | ns | Both IC and VC centered at 0 |
| dark_neuron | ns | Visible neurons: wide scatter; dark: tight at 0 |
| EAP visibility | ns | No difference |

None of the metadata variables explain which cells show the post-spike exponent coupling and which don't. The 40% yield is distributed without obvious structure across cell types, recording modalities, and cortical depths.

This is a genuinely open question. The cells that show the log_ISI × post-spike exponent coupling are not distinguishable from those that don't based on any available metadata — which either means the effect reflects idiosyncratic biology at the cell level, or requires a variable we haven't measured (e.g., firing rate regime, network connectivity, laminar position with finer resolution).

---

## Summary

| Finding | Strength |
|---------|----------|
| PC cells show stronger LFP-spike coupling than IN cells | **Strong** — significant for 4/5 LFP features, BH-FDR corrected |
| Recording modality / clamp mode predicts LFP coupling strength | **None** — all ns |
| Dark neuron / EAP visibility predicts LFP coupling strength | **None** — all ns |
| Cortical depth predicts LFP coupling strength | **None** — ρ ≈ 0 |
| Metadata predicts which spike-LFP feature combo is strongest (3-way) | **None** — all ns after BH-FDR |
| Effect direction (sign of Cohen's d) is explained by metadata | **None** — all groups positive, no reversal |
| Any metadata predicts the 40% log_ISI × post-spike exponent yield | **None** — all ns |
