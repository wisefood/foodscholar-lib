"""Put `research/` on sys.path so the archived bake-off package imports as
`bakeoff` (it was relocated out of `foodscholar.layer_a.bakeoff`). This keeps the
provenance runnable via `pytest research/` without shipping it in the package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
