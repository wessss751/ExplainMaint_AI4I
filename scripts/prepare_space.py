"""Create a reviewed local upload folder; never authenticates or publishes."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT_FILES = (
    'Dockerfile', '.dockerignore', '.gitignore', 'start.sh', 'app.py', 'api.py',
    'core.py', 'report.py', 'requirements.txt', 'requirements-dev.txt', 'requirements-lock.txt',
    'README.md', 'README_HUGGINGFACE.md', 'CHECKLIST_DEPOT.md',
    'one_pager_final.pdf', 'one_pager_modele.pdf', 'LICENSE', '.gitattributes',
)
TREES = ('.streamlit', '.github', 'deployment', 'scripts', 'assets', 'notebooks', 'tests', 'results', 'docs')
SUFFIXES = {'.py', '.md', '.txt', '.toml', '.conf', '.sh', '.json', '.csv',
            '.png', '.jpg', '.jpeg', '.svg', '.ipynb', '.pdf', '.yml', '.yaml', '.xml'}
EXCLUDED_PARTS = {'__pycache__', '.ipynb_checkpoints', '.pytest_cache', '.git', '.env'}
SECRET_NAMES = {'secrets.toml', 'credentials.json', 'tokens.json', '.netrc', '.git-credentials'}


def allowed_file(path, source):
    parts = {part.lower() for part in path.relative_to(source).parts}
    return (path.is_file() and path.suffix.lower() in SUFFIXES
            and not EXCLUDED_PARTS.intersection(parts)
            and not SECRET_NAMES.intersection(parts)
            and not any(part.startswith('.env.') for part in parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('dist/huggingface-space'))
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    if destination == source or source.is_relative_to(destination):
        parser.error('Output must be a separate directory, not the project or its parent')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        parser.error('Output must be empty; choose a new directory to preserve previous packages')
    readme = (source / 'README.md').read_text(encoding='utf-8')
    if not readme.startswith('---\n'):
        parser.error('README.md must start with Hugging Face metadata (sdk: docker; app_port: 7860)')
    metadata = readme.split('---', 2)[1]
    if 'sdk: docker' not in metadata or 'app_port: 7860' not in metadata:
        parser.error('README.md metadata must specify sdk: docker and app_port: 7860')
    selected = [source / name for name in ROOT_FILES if (source / name).is_file()]
    for tree in TREES:
        folder = source / tree
        if folder.is_dir():
            selected.extend(path for path in folder.rglob('*') if allowed_file(path, source))
    for path in selected:
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError(f'Refusing symbolic link or external file: {path.relative_to(source)}')
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sorted(selected):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest.append({'path': relative.as_posix(), 'bytes': path.stat().st_size,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (destination / 'DEPLOYMENT_MANIFEST.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(manifest)} files in {destination}')
    print('No authentication, upload or publication was performed.')


if __name__ == '__main__':
    main()
