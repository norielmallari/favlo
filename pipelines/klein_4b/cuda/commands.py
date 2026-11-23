"""``setup``, ``quantize``, and ``kernels`` for FLUX.2 [klein] 4B."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

from klein_4b.cuda.constants import REPO, SOURCE_BYTES, repo_root, source_dir


def main(argv: Sequence[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("commands: setup, quantize, kernels")
        return 0
    command = argv[0]
    if command == "setup":
        return setup()
    if command == "quantize":
        from klein_4b.cuda.record import quantize

        quantize()
        return 0
    if command == "kernels":
        from favlo.smoke_lock import hold, release

        root = repo_root()
        created = hold(root=root, pipeline="klein_4b", command="kernels")
        try:
            from klein_4b.cuda.checks import run

            run()
        finally:
            if created:
                release(root=root)
        return 0
    print("commands: setup, quantize, kernels")
    return 2


def setup() -> int:
    folder = source_dir()
    if _ready(folder):
        print(folder)
        return 0
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uvx", "hf", "download", REPO, "--local-dir", str(folder)],
        check=True,
    )
    return 0


def _ready(folder) -> bool:
    for relative, size in SOURCE_BYTES.items():
        path = folder / relative
        if not path.is_file() or path.stat().st_size != size:
            return False
    return True
