"""Execute a notebook in-place via nbclient.
Usage: python _run_nb.py NBx.ipynb [per_cell_timeout_s]"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1])
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
nb = nbformat.read(path, as_version=4)
client = NotebookClient(
    nb, timeout=timeout, kernel_name="python3",
    resources={"metadata": {"path": str(path.parent)}},
)
client.execute()
nbformat.write(nb, path)
print(f"OK executed {path.name} ({len(nb.cells)} cells)")
