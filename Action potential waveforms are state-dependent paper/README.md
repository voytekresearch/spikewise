# Action potential waveforms are state-dependent

Martin-Burgos, B., Juavinett, A., Riviere, P., Hammonds, R., & Voytek, B. (2026). Action potential waveforms are state-dependent. *bioRxiv*. https://www.biorxiv.org/content/10.64898/2026.09.15.751814v1

## About this paper

For more than a century, neuroscience has treated the action potential as a binary, all or none event: a neuron either fires or it does not, and every spike is treated as functionally identical to the next. This abstraction underlies most of systems and computational neuroscience, and it even shaped the binary artificial neurons that founded modern AI in the 1940s and 50s. Modern AI abandoned binary neurons decades ago in favor of smooth, graded activation functions, because binary units simply do not learn as well. It turns out the biology may never have been binary in the first place.

This paper asks whether that same binary abstraction has been discarding real, usable information in biological neurons. Using `spikewise`, an open source framework built for this project, individual intracellular action potentials are parameterized into a small set of physiologically interpretable shape features rather than reduced to a single yes/no event. Across two independent datasets, the shape of a single spike, not just whether or when it occurred, turns out to carry real information: it varies systematically with the strength and pattern of the current driving the neuron, it varies from spike to spike within the same neuron more than expected, and it tracks the surrounding local field potential (LFP) state the neuron is embedded in. Waveform shape also predicts the timing of the next spike: faster repolarization goes with a shorter gap before the following spike.

Our results suggest that action potential waveforms are state-dependent. By treating every spike as a binary event, we may be removing critical information and constraining the possible space of neural codes. Instead, each spike may carry analog information about the state that produced it.

## The five main findings

1. Parameterization captures intra-spike correlations
2. Electrical stimulation causally influences spike waveform
3. Neurons show multimodal within-cell spike waveform spontaneous variability
4. Within-neuron waveform variability can exceed between-neuron differences
5. AP waveform predicts peri-spike LFP state

## Repository structure

```
Action potential waveforms are state-dependent paper/
├── 1_parameterization_captures_intra_spike_correlations.ipynb
├── 2_electrical_stimulation_causally_influences_spike_waveform.ipynb
├── 3_neurons_show_multimodal_spontaneous_variability.ipynb
├── 4_within_neuron_variability_exceeds_between_neuron_differences.ipynb
├── 5_ap_waveform_predicts_peri_spike_lfp_state.ipynb
│   # one notebook per main finding: a plain-language walkthrough of the finding, its
│   # published figure, and the exact real function + cached data that generated it
│
├── finding_notebooks_data/
│   # small, post feature extraction data the finding notebooks load, so they run
│   # without downloading any raw recordings (build_*.py scripts show how it was made)
│
├── finding_notebooks_functions/
│   # helper functions used by the finding notebooks (e.g. the interactive cell picker)
│
├── requirements.txt
│   # extra packages needed to run the finding notebooks
│
└── AP_empirical_paper_all_analyses/
    # the full, detailed per-dataset and per-cell analyses behind every figure, plus
    # all supplementary analyses (see its own README.md for the dataset breakdown)
```

## Running the finding notebooks

From the repo root:

```bash
pip install -e .
pip install -r "Action potential waveforms are state-dependent paper/requirements.txt"
```

Then open any of the five notebooks and run all cells. The interactive widgets need a live kernel, so they will not respond in a static preview (e.g. on GitHub).
