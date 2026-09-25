"""Replace saved ipywidgets outputs in a notebook with static snapshots of what they were showing.

Widget outputs cannot be drawn until the notebook is run, so a saved notebook shows a
"could not render" warning in their place. This keeps whatever the widget's plot area was
displaying (figures, printed text) as ordinary outputs, so the notebook looks complete when
opened; running the cell brings the interactive controls back.

Usage: python freeze_widget_outputs.py NOTEBOOK.ipynb [NOTEBOOK2.ipynb ...]
"""
import json
import sys

WIDGET_MIME = 'application/vnd.jupyter.widget-view+json'
NOTE = '(Static snapshot. Run this cell to use the interactive controls.)\n'


def freeze(path):
    with open(path) as fh:
        nb = json.load(fh)
    state = nb.get('metadata', {}).get('widgets', {}).get('application/vnd.jupyter.widget-state+json', {}).get('state', {})
    n_frozen = 0
    for cell in nb['cells']:
        outputs = cell.get('outputs', [])
        if not any(WIDGET_MIME in o.get('data', {}) for o in outputs):
            continue
        frozen = [{'output_type': 'stream', 'name': 'stdout', 'text': [NOTE]}]
        for o in outputs:
            if WIDGET_MIME not in o.get('data', {}):
                frozen.append(o)
                continue
            model = state.get(o['data'][WIDGET_MIME]['model_id'], {})
            if model.get('model_name') == 'OutputModel':
                frozen.extend(model['state'].get('outputs', []))
        cell['outputs'] = frozen
        n_frozen += 1
    nb.get('metadata', {}).pop('widgets', None)
    with open(path, 'w') as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write('\n')
    print(f'{path}: froze {n_frozen} widget cell(s)')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        freeze(p)
