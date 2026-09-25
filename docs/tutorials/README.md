# spikewise tutorials

`spikewise` was developed for, and is described in:

> Martin-Burgos, B., Juavinett, A., Riviere, P., Hammonds, R., & Voytek, B. (2026). *Action potential waveforms are state-dependent*. bioRxiv. https://www.biorxiv.org/content/10.64898/2026.09.15.751814v1

The paper's analyses, with one walkthrough notebook per main finding, are in [`Action potential waveforms are state-dependent paper/`](../../Action%20potential%20waveforms%20are%20state-dependent%20paper/).

| notebook | what it covers |
|---|---|
| [01_quickstart.ipynb](01_quickstart.ipynb) | Fit the spikes in one recording with `Spike`, plot the fits, read the feature table, filter by fit quality, and the main settings (including `flip_signal` for recordings where spikes point down) |
| [02_many_recordings.ipynb](02_many_recordings.ipynb) | Fit many sweeps at once with `SpikeGroup`, then compare features across conditions and look at how features correlate |
| [03_spikes_and_other_signals.ipynb](03_spikes_and_other_signals.ipynb) | Use `.alt()` to describe a second signal (here the injected current) around each spike, and relate it to spike shape |

All three run on small excerpts of two open datasets bundled in `data/example_recordings.npz` (2.6 MB), so nothing needs to be downloaded. `data/build_tutorial_data.py` shows how the excerpts were made.

From the repo root:

```bash
pip install -e .
pip install jupyter seaborn scikit-learn   # used by tutorials 2 and 3
```
