"""
Patch all spe-1_c*_clusters.ipynb notebooks to:
  1. Add / set FORCE_REFIT = True in the run-control flags cell
  2. Wrap sp = Spike(...) in a pickle load-or-fit conditional
  3. Wrap sp.fit(...) in the same conditional
  4. Wrap sp.filter_features() in the conditional, re-add R² cols, save pickle
  5. Insert a KDE R² distribution plot cell right after filter_features

Idempotent: already-patched cells are detected and skipped (c1 was patched manually).
"""

import nbformat
import glob
import os
import re
import uuid

NB_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'

R2_PLOT_SOURCE = (
    "from scipy.stats import gaussian_kde as _kde\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "for ax, r2_arr, title in zip(\n"
    "        axes,\n"
    "        [sp.r_squared_exp, sp.r_squared_ramp],\n"
    "        ['Exp decay R²', 'Ramp fit R²']):\n"
    "    vals = np.asarray(r2_arr)\n"
    "    vals = vals[np.isfinite(vals)]\n"
    "    if len(vals) >= 5:\n"
    "        try:\n"
    "            kde_fn = _kde(vals)\n"
    "            x_grid = np.linspace(max(0, vals.min() - 0.02), min(1, vals.max() + 0.02), 300)\n"
    "            y_grid = kde_fn(x_grid)\n"
    "            ax.fill_between(x_grid, y_grid, alpha=0.4, color='steelblue')\n"
    "            ax.plot(x_grid, y_grid, color='steelblue', lw=2)\n"
    "        except Exception:\n"
    "            pass\n"
    "    med = np.median(vals)\n"
    "    ax.axvline(med, color='#333', lw=1.5, ls='--', label=f'median = {med:.3f}')\n"
    "    ax.set_title(f'c{cell_num}  {title}', fontsize=14, fontweight='bold')\n"
    "    ax.set_xlabel(title, fontsize=12)\n"
    "    ax.set_ylabel('Density', fontsize=12)\n"
    "    ax.set_xlim(0, 1)\n"
    "    ax.legend(fontsize=11)\n"
    "    sns.despine(ax=ax)\n"
    "plt.suptitle(f'Spike fit R² distributions — cell {cell_num}', fontsize=14, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

FILTER_SOURCE = (
    "if not _loaded_from_pickle:\n"
    "    sp.filter_features()\n"
    "    # Re-add R² columns (filter_features drops them by default)\n"
    "    sp.df_features['r_squared_exp']  = sp.r_squared_exp\n"
    "    sp.df_features['r_squared_ramp'] = sp.r_squared_ramp\n"
    "    with open(spike_fit_pickle, 'wb') as _fh:\n"
    "        pickle.dump(sp, _fh)\n"
    "    print(f'Saved spike fit → {os.path.basename(spike_fit_pickle)}')\n"
    "else:\n"
    "    if 'r_squared_exp' not in sp.df_features.columns:\n"
    "        sp.df_features['r_squared_exp']  = sp.r_squared_exp\n"
    "        sp.df_features['r_squared_ramp'] = sp.r_squared_ramp\n"
    "    print(f'[cache] {len(sp.df_features)} spikes loaded')"
)

patched = []
skipped = []

for nb_path in sorted(glob.glob(NB_DIR + 'spe-1_c*_clusters.ipynb')):
    nb = nbformat.read(nb_path, as_version=4)
    modified = False
    filter_cell_idx = None
    has_r2_plot = False

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        src = cell.source

        # Detect already-inserted R² plot
        if 'gaussian_kde' in src and 'r_squared_exp' in src and 'r_squared_ramp' in src:
            has_r2_plot = True

        # ── 1. Run-control flags ─────────────────────────────────────────
        if 'FORCE_RERUN' in src:
            if 'FORCE_REFIT' not in src:
                cell.source = src.rstrip() + '\nFORCE_REFIT  = True    # set True to refit spikes (ignores existing spike fit pickle)'
                modified = True
            elif re.search(r'FORCE_REFIT\s*=\s*False', src):
                cell.source = re.sub(r'FORCE_REFIT\s*=\s*False', 'FORCE_REFIT  = True ', src)
                modified = True
            # already True → no change needed

        # ── 2. Spike init ────────────────────────────────────────────────
        if 'sp = Spike(' in src and 'spike_fit_pickle' not in src:
            spike_line = next(
                (l.strip() for l in src.split('\n') if 'sp = Spike(' in l), None
            )
            if spike_line:
                cell.source = (
                    "spike_fit_pickle = os.path.join(SPE1_PICKLE_ROOT, 'spike_fit_pickles', f'c{cell_num}_spike_fit.pkl')\n"
                    "os.makedirs(os.path.dirname(spike_fit_pickle), exist_ok=True)\n"
                    "\n"
                    "if not FORCE_REFIT and os.path.exists(spike_fit_pickle):\n"
                    "    print(f'[cache] Loading spike fit from {os.path.basename(spike_fit_pickle)}')\n"
                    "    with open(spike_fit_pickle, 'rb') as _fh:\n"
                    "        sp = pickle.load(_fh)\n"
                    "    _loaded_from_pickle = True\n"
                    "else:\n"
                    f"    {spike_line}\n"
                    "    _loaded_from_pickle = False"
                )
                modified = True

        # ── 3. sp.fit ────────────────────────────────────────────────────
        if 'sp.fit(' in src and '_loaded_from_pickle' not in src:
            indented = '\n'.join('    ' + l for l in src.strip().split('\n'))
            cell.source = 'if not _loaded_from_pickle:\n' + indented
            modified = True

        # ── 4. filter_features ───────────────────────────────────────────
        if 'sp.filter_features()' in src and '_loaded_from_pickle' not in src:
            cell.source = FILTER_SOURCE
            filter_cell_idx = i
            modified = True

    # ── 5. Insert R² plot cell after filter cell ─────────────────────────
    if filter_cell_idx is not None and not has_r2_plot:
        r2_cell = nbformat.v4.new_code_cell(source=R2_PLOT_SOURCE)
        r2_cell['id'] = uuid.uuid4().hex[:8]
        nb.cells.insert(filter_cell_idx + 1, r2_cell)
        modified = True

    if modified:
        nbformat.write(nb, nb_path)
        patched.append(os.path.basename(nb_path))
    else:
        skipped.append(os.path.basename(nb_path))

print(f'Patched  {len(patched)} notebooks:')
for nb in patched:
    print(f'  ✓ {nb}')
if skipped:
    print(f'\nSkipped (already up-to-date) {len(skipped)}:')
    for nb in skipped:
        print(f'  - {nb}')
