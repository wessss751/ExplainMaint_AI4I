"""Dependency-free checks for the source delivery and protected original files."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []
    manifest = json.loads((ROOT / 'docs/original_artifacts.sha256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f'Original artifact changed or missing: {name}')
    sources = list(ROOT.glob('*.py'))
    for directory in ('scripts', 'tests', 'deployment'):
        sources.extend((ROOT / directory).rglob('*.py'))
    for source in sources:
        try:
            ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
        except SyntaxError as exc:
            failures.append(str(exc))
    notebook = json.loads((ROOT / 'notebooks/analyse_ai4i.ipynb').read_text(encoding='utf-8'))
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    for index, cell in enumerate(cells):
        try:
            ast.parse(''.join(cell['source']), filename=f'notebook cell {index + 1}')
        except SyntaxError as exc:
            failures.append(str(exc))
    for name in ('start.sh', 'Dockerfile', 'deployment/nginx.conf'):
        if b'\r\n' in (ROOT / name).read_bytes():
            failures.append(f'{name} must use LF for Linux')
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    if not readme.startswith('---\n') or 'sdk: docker' not in readme.split('---', 2)[1]:
        failures.append('README needs Hugging Face Docker metadata')
    print(f'Checked {len(manifest)} original artifacts, {len(sources)} Python files and {len(cells)} notebook cells.')
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        raise SystemExit(1)
    print('PASS: original files unchanged, source syntax valid, Linux line endings valid.')
    print('These static checks do not execute ML models or prove deployment availability.')


if __name__ == '__main__':
    main()
