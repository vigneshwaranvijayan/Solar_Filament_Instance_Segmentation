"""Check or refresh the notebook's embedded pipeline sources using standard Python."""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Update embedded sources and clear execution outputs')
    args = parser.parse_args()
    path = ROOT / 'V5.ipynb'
    nb = json.loads(path.read_text())
    cell = next(c for c in nb['cells'] if c['cell_type'] == 'code' and 'SOURCES = ' in ''.join(c['source']))
    text = ''.join(cell['source'])
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in n.targets))
    embedded = ast.literal_eval(node.value)
    actual = {name: (ROOT / name).read_text() for name in embedded}
    for name, source in actual.items():
        compile(source, name, 'exec')
    if args.write:
        start = text.index('SOURCES = ')
        end = text.index('\nfor name, text in SOURCES.items():', start)
        cell['source'] = (text[:start] + 'SOURCES = ' + repr(actual) + text[end:]).splitlines(True)
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                c['execution_count'] = None
                c['outputs'] = []
        path.write_text(json.dumps(nb, indent=1))
        print('Updated V5.ipynb with', len(actual), 'pipeline source files.')
    else:
        differences = [name for name in actual if actual[name] != embedded[name]]
        if differences:
            raise SystemExit('Notebook differs from: ' + ', '.join(differences) + '. Run with --write after reviewing your edits.')
        print('Notebook and', len(actual), 'pipeline source files match.')
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), f'notebook cell {i}', 'exec')


if __name__ == '__main__':
    main()
