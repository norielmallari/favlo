"""Commands for ``favlo --wan22_ti2v_5b`` on CUDA."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from wan22_ti2v_5b.cuda.layout import BUDGET, dit_bytes, longest, opening_bytes, umt5_bytes

PIPELINE = "wan22_ti2v_5b"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"favlo --{PIPELINE}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="print the 6 GiB openings")
    sub.add_parser("kernels", help="compile the CUDA step and run its checks")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "plan":
            _plan()
            return 0
        if args.command == "kernels":
            from wan22_ti2v_5b.cuda.checks import run_kernels

            run_kernels()
            return 0
    except (OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 2


def _plan() -> None:
    print(f"budget {BUDGET}")
    print(f"dit {dit_bytes()}")
    print(f"umt5 {umt5_bytes()}")
    for height, width in ((704, 1280), (480, 832), (320, 576)):
        frames, _time = longest(height, width)
        print(f"clip {width}x{height} frames {frames} bytes {opening_bytes(frames, height, width)}")
