from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    # tools/ -> repo root
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root = str(_repo_root())
    # Put repo root BEFORE tools/ so `import prov_qa` resolves to the package folder `prov_qa/`
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repo_on_path()

# If Python accidentally loaded a non-package `prov_qa` earlier, remove it.
m = sys.modules.get("prov_qa")
if m is not None and not hasattr(m, "__path__"):
    del sys.modules["prov_qa"]

from prov_qa.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
