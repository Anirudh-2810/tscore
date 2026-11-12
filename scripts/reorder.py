#!/usr/bin/env python3
"""Reorder math + guards for ShopBrain (plan Week 3).

Turns a per-item demand forecast into suggested reorder quantities that
ALWAYS respect the owner's limits:

  line level   need      = ceil(forecast * lead_time * (1 + safety_buffer))
                          - on_hand
               guard A   never exceed max_order_qty
               guard B   never exceed last_order_qty * max_multiple_of_last
  week level   guard C   total spend never exceeds budget_cap — lines are
                         filled most-urgent-first (lowest weeks-of-cover),
                         the binding line may be partially filled, the rest
                         get zero.

Every function here is pure — StockOffline's report (Week 4) calls these;
tests in tests/test_reorder.py prove the limits hold over an input grid.

Usage:
  python scripts/reorder.py --demo
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class Line:
    """One reorder line: forecast and shop state for an item."""
    item: str
    forecast: float            # predicted demand for the next lead-time window
    on_hand: int
    unit_price: float
    last_order_qty: Optional[int] = None   # None = never ordered before


@dataclass
class Guards:
    """Owner-set limits. None/0 disables that guard (budget: None only)."""
    safety_buffer: float = 0.20           # +20% over the lead-time need
    lead_time_weeks: int = 1              # delivery delay covered per order
    max_order_qty: Optional[int] = None   # absolute per-line cap
    max_multiple_of_last: Optional[float] = None  # qty <= last * this
    budget_cap: Optional[float] = None    # total spend cap for the week


@dataclass
class Result:
    item: str
    forecast: float
    on_hand: int
    qty: int
    spend: float
    capped_by: List[str] = field(default_factory=list)  # which guards fired


def raw_need(forecast: float, on_hand: int, guards: Guards) -> int:
    """Uncapped reorder: lead-time demand + safety buffer, minus stock."""
    if forecast < 0 or on_hand < 0:
        raise ValueError("forecast/on_hand must be >= 0")
    if guards.lead_time_weeks < 1:
        raise ValueError("lead_time_weeks must be >= 1")
    if guards.safety_buffer < 0:
        raise ValueError("safety_buffer must be >= 0")
    need = math.ceil(forecast * guards.lead_time_weeks
                     * (1.0 + guards.safety_buffer)) - on_hand
    return max(0, need)


def suggested_qty(line: Line, guards: Guards) -> Result:
    """Per-line quantity after guard A (max order) and guard B (last-order
    multiple). Budget is enforced across lines in allocate()."""
    qty = raw_need(line.forecast, line.on_hand, guards)
    capped: List[str] = []
    if line.unit_price < 0:
        raise ValueError("unit_price must be >= 0")

    if guards.max_order_qty is not None:
        if guards.max_order_qty < 0:
            raise ValueError("max_order_qty must be >= 0")
        if qty > guards.max_order_qty:
            qty = guards.max_order_qty
            capped.append("max_order_qty")

    if guards.max_multiple_of_last is not None:
        if guards.max_multiple_of_last <= 0:
            raise ValueError("max_multiple_of_last must be > 0")
        if line.last_order_qty is not None:
            ceiling = math.floor(line.last_order_qty * guards.max_multiple_of_last)
            if qty > ceiling:
                qty = ceiling
                capped.append("max_multiple_of_last")

    return Result(item=line.item, forecast=line.forecast, on_hand=line.on_hand,
                  qty=qty, spend=qty * line.unit_price, capped_by=capped)


def weeks_of_cover(line: Line) -> float:
    """Weeks of stock left — lower is more urgent."""
    if line.forecast <= 0:
        return math.inf if line.on_hand > 0 else 0.0
    return line.on_hand / line.forecast


def allocate(lines: Sequence[Line], guards: Guards) -> List[Result]:
    """All lines through per-line guards, then the budget (guard C).

    Fill order: most urgent first (lowest weeks-of-cover, name tie-break).
    If the binding line can't fit whole, it is partially filled with what
    the budget allows; every later line gets zero.
    Returns results in the original input order.
    """
    if guards.budget_cap is not None and guards.budget_cap < 0:
        raise ValueError("budget_cap must be >= 0")
    names = [l.item for l in lines]
    if len(set(names)) != len(names):
        raise ValueError("duplicate item names in reorder lines")

    results = {r.item: r for r in (suggested_qty(l, guards) for l in lines)}

    if guards.budget_cap is not None:
        remaining = guards.budget_cap
        by_urgency = sorted(lines, key=lambda l: (weeks_of_cover(l), l.item))
        for line in by_urgency:
            res = results[line.item]
            if res.spend <= remaining + 1e-9:
                remaining -= res.spend
                continue
            affordable = int(math.floor((remaining + 1e-9) / line.unit_price)) \
                if line.unit_price > 0 else 0
            if res.qty > affordable:
                res.qty = affordable
                res.spend = affordable * line.unit_price
                res.capped_by.append("budget_cap")
            remaining = 0.0
    return [results[l.item] for l in lines]


# -- demo -----------------------------------------------------------------
def _demo() -> int:
    guards = Guards(safety_buffer=0.25, lead_time_weeks=2,
                    max_order_qty=60, max_multiple_of_last=3.0,
                    budget_cap=5000.0)
    lines = [
        Line("PARACETAMOL-500", forecast=80, on_hand=10, unit_price=30.0,
             last_order_qty=40),
        Line("INSULIN-PEN-ND", forecast=20, on_hand=4, unit_price=150.0,
             last_order_qty=10),
        Line("VITAMIN-C-1000", forecast=15, on_hand=120, unit_price=25.0),
        Line("BP-MONITOR-CUFF", forecast=6, on_hand=50, unit_price=400.0,
             last_order_qty=5),
    ]
    print("Reorder demo — guards:"
          f" max_order={guards.max_order_qty},"
          f" <= {guards.max_multiple_of_last}x last order,"
          f" budget={guards.budget_cap},"
          f" safety=+{guards.safety_buffer:.0%},"
          f" lead={guards.lead_time_weeks}w")
    print(f"{'item':<22}{'fcst':>6}{'hand':>6}{'qty':>6}{'spend':>10}  caps")
    total = 0.0
    for r in allocate(lines, guards):
        total += r.spend
        print(f"{r.item:<22}{r.forecast:>6.0f}{r.on_hand:>6}{r.qty:>6}"
              f"{r.spend:>10.2f}  {','.join(r.capped_by) or '-'}")
    print(f"{'TOTAL':<22}{'':>18}{total:>10.2f}  (budget {guards.budget_cap:.2f})")
    assert total <= guards.budget_cap + 1e-9, "budget violated"
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", action="store_true",
                    help="run the bundled example (report UI comes in Week 4)")
    args = ap.parse_args(argv)
    if args.demo:
        return _demo()
    ap.error("no action — try --demo (full report lands in Week 4)")


if __name__ == "__main__":
    sys.exit(main())
