"""Validate notebook syntax; optionally execute every cell with real dependencies.

Run from the repository root:
    python scripts/validate_notebook.py
    python scripts/validate_notebook.py --execute

The executed notebook replaces its own outputs only after complete success.
Historical one-pager PDFs and screenshots are never written by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "analyse_ai4i.ipynb"


def protected_files() -> dict[Path, str]:
    paths = [ROOT / "one_pager_final.pdf", ROOT / "one_pager_modele.pdf"]
    paths.extend(sorted((ROOT / "assets").rglob("*")))
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute all cells and save genuine outputs")
    args = parser.parse_args()
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    count = 0
    identifiers: set[str] = set()
    for index, cell in enumerate(document["cells"], start=1):
        cell_id = cell.get("id")
        if cell_id in identifiers:
            raise ValueError(f"Duplicate cell id: {cell_id}")
        identifiers.add(cell_id)
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK.name}:cell-{index}", "exec")
            count += 1
    print(f"Notebook JSON and Python syntax valid: {count} code cells.")
    if not args.execute:
        print("Static validation only; no ML result generated. Use --execute for runtime validation.")
        return

    try:
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import AsyncKernelManager
    except ImportError as exc:
        raise SystemExit("Install requirements-dev.txt in this Python environment before --execute.") from exc

    before = protected_files()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    try:
        with TemporaryDirectory(prefix=".notebook-runtime-", dir=ROOT) as runtime:
            manager = AsyncKernelManager(
                kernel_name="python3", connection_file=str(Path(runtime) / "kernel.json"),
            )
            NotebookClient(
                notebook, timeout=900, kernel_name="python3", km=manager,
                resources={"metadata": {"path": str(ROOT)}},
            ).execute(cleanup_kc=True)
    finally:
        if protected_files() != before:
            raise RuntimeError("A protected historical PDF or asset changed during execution")
    errors = [
        output for cell in notebook.cells if cell.cell_type == "code"
        for output in cell.get("outputs", []) if output.output_type == "error"
    ]
    if errors:
        raise RuntimeError("Notebook contains execution errors; outputs were not saved")
    nbformat.write(notebook, NOTEBOOK)
    print(f"All {count} code cells executed; genuine outputs saved to {NOTEBOOK.relative_to(ROOT)}.")
    print("Historical PDFs and assets are unchanged (SHA-256 verified).")


if __name__ == "__main__":
    main()
