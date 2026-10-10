#!/usr/bin/env python3
"""Generate the ExBind-v2 broad diagnostic suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exbind.evaluation.cross_format import run_cross_format_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--n-per-family", type=int, default=50)
    parser.add_argument("--n-cross-format-groups", type=int, default=50)
    args = parser.parse_args()
    result = run_cross_format_benchmark(
        out_dir=args.out,
        n_per_family=args.n_per_family,
        n_cross_format_groups=args.n_cross_format_groups,
        seed=args.seed,
        surface_variant="interface_gate_v1",
    )
    print({"records": len(result["records"]), "groups": len(result["groups"])})


if __name__ == "__main__":
    main()
