#!/usr/bin/env python3
"""Week 8 - one-command demo of the whole ShopBrain offline pipeline.

Chains every tscore script on seeded fake data so a screen recording (or a
skeptical shop owner) can watch the full story in one run:

  1. fake sales  - seeded generator (14-week + 52-week series)
  2. backtest    - walk-forward evaluation + naive gate verdict
  3. report      - report.txt + forecast.csv (what a shop gets each week)
  4. agents      - Week 7 live-and-die replay dashboard (open + press Play)

The Week 6 RNN test is internal QA (verdict: DROP) and is NOT part of the
shop-facing demo - pass --with-rnn to include it.

No real shop data is used or produced; every artifact lands under examples/
or out-demo/. Console output is ASCII only. Exit 0 = all steps ran, 1 = a
step failed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
EXAMPLES = ROOT / "examples"


def steps(with_rnn: bool = False) -> List[Tuple[str, List[str], Tuple[int, ...]]]:
    """(title, argv, acceptable exit codes) for each demo stage.

    argv[0] is a script path relative to the repo root; run() prefixes it
    with sys.executable. RNN accepts exit 2 because DROP is its honest,
    expected verdict (see docs/w6-rnn-verdict.md).
    """
    chain: List[Tuple[str, List[str], Tuple[int, ...]]] = [
        ("1/4 fake sales (seeded, fixed start so it is byte-identical "
         "any day)",
         ["scripts/generate_fake_sales.py", "--out", "examples/fake_sales.csv",
          "--weeks", "13", "--seed", "42", "--start", "2026-08-12"],
         (0,)),
        ("1/4 fake sales, 52 weeks (for the agent replay)",
         ["scripts/generate_fake_sales.py", "--out",
          "examples/fake_sales_52w.csv", "--weeks", "52", "--seed", "42",
          "--start", "2026-01-05"],
         (0,)),
        ("2/4 walk-forward backtest + gate verdict",
         ["scripts/backtest.py", "--sales", "examples/fake_sales.csv"],
         (0,)),                     # gate passes on the fake data (WAPE 28.10%)
        ("3/4 weekly report (report.txt + forecast.csv)",
         ["scripts/report.py", "--sales", "examples/fake_sales.csv",
          "--stock", "examples/stock_example.csv", "--out-dir", "out-demo"],
         (0,)),
        ("4/4 live-and-die agents (log + replay dashboard)",
         ["scripts/agents_sim.py", "--sales", "examples/fake_sales_52w.csv",
          "--html", "examples/agents_dashboard.html",
          "--log", "examples/agents_run.log"],
         (0,)),
    ]
    if with_rnn:
        chain.insert(3, (
            "3/4b Week 6 RNN test (expect DROP, exit 2)",
            ["scripts/rnn_test.py", "--sales", "examples/fake_sales.csv"],
            (0, 2)))
    return chain


def run(argv: Sequence[str], ok_exits: Tuple[int, ...],
        dry_run: bool = False) -> bool:
    cmd = [sys.executable] + list(argv)
    print(f"\n--- {' '.join(argv)}")
    if dry_run:
        return True
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode in ok_exits:
        return True
    print(f"demo step FAILED: exit {proc.returncode} "
          f"(expected one of {ok_exits})", file=sys.stderr)
    return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--with-rnn", action="store_true",
                    help="also run the Week 6 RNN test (internal QA)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    args = ap.parse_args(argv)

    for title, step_argv, ok_exits in steps(args.with_rnn):
        print(f"\n=== {title}")
        if not run(step_argv, ok_exits, dry_run=args.dry_run):
            return 1

    print("\n" + "=" * 60)
    print("DEMO COMPLETE - artifacts:")
    print("  out-demo/report.txt        shop-readable weekly report")
    print("  out-demo/forecast.csv      contract forecast for the app")
    print("  examples/agents_dashboard.html  open in a browser, press Play")
    print("  examples/agents_run.log    full event log (deaths/spawns)")
    print("Gate/backtest verdicts are printed inline above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
