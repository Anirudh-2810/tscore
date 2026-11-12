#!/usr/bin/env python3
"""Walk-forward backtest of ShopBrain baselines on a contract sales CSV.

Models (weekly grain, per item):
  naive_last_week     GATE - "reorder what sold last week":  y_hat[t] = y[t-1]
  seasonal_naive      same week one month ago (lag 4 weeks) - stand-in seasonal
                       until 52 weeks of data exist (documented substitution)
  moving_average(k)   mean of the last k weeks (k via --ma-k, default 2,4,8)

Every model is scored on identical folds: expanding window, predict the next
week from all history up to that point, then roll forward - nothing from the
future can leak into a prediction. Stockouts come from a per-item carry-over
simulation: the week's order is placed from the forecast and is available
before that week's demand (lead time folded into the weekly grain; a
production version adds explicit lead-time days).

Gate (standing rule 1 of the ShopBrain plan): at least one baseline must beat
naive_last_week on WAPE or forecasting work does NOT proceed.

Exit codes: 0 = gate passed, 2 = gate failed, 1 = input/usage error.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

GATE = "naive_last_week"
SALES_COLUMNS = ("item", "date", "quantity")

Hist = List[float]
Forecaster = Callable[[Hist], float]


# -- input ----------------------------------------------------------------
def load_sales(path: str) -> List[Tuple[str, date, int]]:
    """Read a contract sales CSV. Hard error on missing columns; malformed
    rows are skipped with a warning (same philosophy as StockOffline import).
    """
    rows: List[Tuple[str, date, int]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [(h or "").strip() for h in (reader.fieldnames or [])]
        missing = [c for c in SALES_COLUMNS if c not in header]
        if missing:
            raise ValueError(
                f"not a contract sales CSV: missing column(s) {', '.join(missing)}")
        key_map = {(raw or "").strip(): raw for raw in (reader.fieldnames or [])}
        skipped = 0
        for lineno, raw in enumerate(reader, start=2):
            row = {name: raw.get(r) for name, r in key_map.items()}
            item = (row.get("item") or "").strip()
            try:
                when = datetime.strptime((row.get("date") or "").strip(),
                                         "%Y-%m-%d").date()
                qty = int(float(row.get("quantity") or ""))
            except (ValueError, TypeError):
                skipped += 1
                continue
            if not item or qty < 1:
                skipped += 1
                continue
            rows.append((item, when, qty))
    if not rows:
        raise ValueError("no usable rows in sales CSV")
    if skipped:
        print(f"warning: skipped {skipped} malformed row(s)", file=sys.stderr)
    return rows


# -- weekly bucketing -----------------------------------------------------
def week_start(d: date) -> date:
    """Monday of d's ISO week (stable week key)."""
    return d - timedelta(days=d.weekday())


def weekly_series(
    rows: Sequence[Tuple[str, date, int]]
) -> Tuple[List[date], Dict[str, List[Optional[float]]]]:
    """Bucket daily rows into calendar weeks.

    Returns (weeks, series):
      weeks   - every Monday from earliest to latest sale (global axis)
      series  - item -> list aligned to weeks; None before the item's first
                sale (pre-introduction weeks are not zeros), 0.0 for
                no-sale weeks after it (an item on the shelf sells zero).
    """
    if not rows:
        raise ValueError("no rows")
    # Dense weekly axis: every Monday from first to last sale. Weeks with no
    # sales AT ALL must still exist, or "last week" would silently mean
    # "two weeks ago" across a gap (shop closed, data outage).
    first_mon = week_start(min(d for _, d, _ in rows))
    last_mon = week_start(max(d for _, d, _ in rows))
    mondays: List[date] = []
    cursor = first_mon
    while cursor <= last_mon:
        mondays.append(cursor)
        cursor += timedelta(weeks=1)
    axis: Dict[date, int] = {w: i for i, w in enumerate(mondays)}
    totals: Dict[str, Dict[int, int]] = {}
    for item, d, qty in rows:
        totals.setdefault(item, {})[axis[week_start(d)]] = \
            totals.setdefault(item, {}).get(axis[week_start(d)], 0) + qty

    series: Dict[str, List[Optional[float]]] = {}
    for item, per_week in totals.items():
        first = min(per_week)
        vals: List[Optional[float]] = [None] * len(mondays)
        for i in range(first, len(mondays)):
            vals[i] = float(per_week.get(i, 0))
        series[item] = vals
    return mondays, series


# -- forecasters ----------------------------------------------------------
def f_naive_last_week(hist: Hist) -> float:
    return hist[-1]


def f_seasonal_naive(lag: int = 4) -> Forecaster:
    def f(hist: Hist) -> float:
        return hist[-(lag + 1)]
    return f


def f_moving_average(k: int) -> Forecaster:
    def f(hist: Hist) -> float:
        return sum(hist[-k:]) / k
    return f


def required_history(models: Sequence[Tuple[str, Forecaster, int]]) -> int:
    """Smallest history all models can score (identical folds for fairness)."""
    return max((need for _, _, need in models), default=1)


# -- evaluation -----------------------------------------------------------
Stats = Dict[str, Dict[str, float]]
Pred = Tuple[str, int, float, float]  # (item, week_index, prediction, actual)


def run_stock(pairs: Sequence[Tuple[float, float]], initial: float
              ) -> int:
    """Carry-over stock simulation. pairs = [(order_from_forecast, demand)].

    Order is available before that week's demand; surplus carries; a week
    whose demand exceeds available stock is a stockout (demand beyond stock
    is lost - the conservative read for a pharmacy).
    Returns the number of stockout weeks.
    """
    stock = float(initial)
    stockouts = 0
    for pred, actual in pairs:
        order = max(0.0, round(pred))
        available = stock + order
        if actual > available:
            stockouts += 1
            stock = 0.0
        else:
            stock = available - actual
    return stockouts


def evaluate(
    weeks: Sequence[date],
    series: Dict[str, List[Optional[float]]],
    models: Sequence[Tuple[str, Forecaster]],
    min_hist: int,
) -> Tuple[Stats, List[Pred]]:
    """Walk-forward: predict week cut+1 from history up to cut, roll forward.

    All models see the same (item, week) folds. Returns (stats, predictions).
    """
    stats: Stats = {
        name: {"ae": 0.0, "se": 0.0, "act": 0.0, "n": 0.0, "under": 0.0,
               "stockouts": 0.0}
        for name, _ in models
    }
    preds: List[Pred] = []
    for item, vals in series.items():
        first = next(i for i, v in enumerate(vals) if v is not None)
        stk = {name: None for name, _ in models}   # lazily initialised
        pairs: Dict[str, List[Tuple[float, float]]] = {n: [] for n, _ in models}
        for cut in range(first, len(weeks) - 1):
            hist_len = cut - first + 1
            if hist_len < min_hist:
                continue
            hist = [v for v in vals[first:cut + 1] if v is not None]
            actual = vals[cut + 1]
            if actual is None:
                continue
            for name, fn in models:
                p = float(fn(hist))
                if stk[name] is None:
                    stk[name] = hist[-1]        # start: last week's sales on hand
                preds.append((item, cut + 1, p, actual))
                s = stats[name]
                s["ae"] += abs(p - actual)
                s["se"] += (p - actual) ** 2
                s["act"] += actual
                s["n"] += 1
                if p < actual:
                    s["under"] += 1
                pairs[name].append((p, actual))
        for name, _ in models:
            if pairs[name]:
                stats[name]["stockouts"] += run_stock(pairs[name], stk[name] or 0.0)
    return stats, preds


# -- metrics + verdict ----------------------------------------------------
def wape(s: Dict[str, float]) -> Optional[float]:
    return (s["ae"] / s["act"]) if s["act"] > 0 else None


def verdict(stats: Stats) -> Tuple[str, Optional[str], bool]:
    """Returns (winner_excluding_gate, reason, gate_passed)."""
    gate = wape(stats[GATE])
    others = {n: wape(s) for n, s in stats.items() if n != GATE}
    others = {n: v for n, v in others.items() if v is not None}
    if not others or gate is None:
        return "", "not enough data to compare", False
    winner = min(others, key=others.get)
    passed = others[winner] < gate
    return winner, None, passed


def report(
    stats: Stats,
    weeks: Sequence[date],
    series: Dict[str, List[Optional[float]]],
    min_hist: int,
    sales_path: str,
) -> Tuple[str, bool]:
    n_items = len(series)
    eval_items = sum(
        1 for v in series.values()
        if sum(1 for x in v if x is not None) >= min_hist
    )
    preds_per_model = int(stats[GATE]["n"])
    out: List[str] = []
    out.append(f"Walk-forward backtest - {sales_path}")
    out.append(
        f"  weeks: {len(weeks)} ({weeks[0]}..{weeks[-1]})  items: {n_items}"
        f" ({eval_items} with >= {min_hist}w history)"
        f"  predictions/model: {preds_per_model}")
    out.append("  assumptions: order for week t arrives before week t's demand;"
               " expanding window; per item")
    out.append("")
    out.append(f"{'model':<24}{'MAE':>8}{'RMSE':>8}{'WAPE%':>8}"
               f"{'under%':>8}{'stockout%':>10}  vs gate")
    for name, s in stats.items():
        n = s["n"]
        if n == 0:
            out.append(f"{name:<24}   (no folds - not evaluated)")
            continue
        mae = s["ae"] / n
        rmse = math.sqrt(s["se"] / n)
        wp = wape(s)
        under = 100.0 * s["under"] / n
        so = 100.0 * s["stockouts"] / n
        gate_w = wape(stats[GATE])
        if name == GATE:
            tag = "gate"
        elif wp is None or gate_w is None:
            tag = "?"
        else:
            tag = "PASS" if wp < gate_w else "FAIL"
        out.append(f"{name:<24}{mae:>8.2f}{rmse:>8.2f}"
                   f"{(100 * wp if wp is not None else float('nan')):>8.2f}"
                   f"{under:>8.1f}{so:>10.1f}  {tag}")
    out.append("")
    winner, reason, passed = verdict(stats)
    gate_w = wape(stats[GATE])
    best_w = wape(stats[winner]) if winner else None
    if passed and winner:
        out.append(f"winner (excl. gate): {winner}")
        out.append(
            f"GATE VERDICT: PASS - {winner} WAPE {100 * best_w:.2f}%"
            f" < naive gate {100 * gate_w:.2f}%")
    else:
        out.append("GATE VERDICT: FAIL - no baseline beats"
                   " naive \"reorder what sold last week\""
                   + (f" (best: {winner} {100 * best_w:.2f}%"
                      f" vs gate {100 * gate_w:.2f}%)" if winner else ""))
        out.append("  -> standing rule 1: fix the data or approach BEFORE"
                   " any model work.")
        if reason:
            out.append(f"  ({reason})")
    return "\n".join(out), passed


# -- main -----------------------------------------------------------------
def build_models(ma_ks: Sequence[int], lag: int) -> List[Tuple[str, Forecaster, int]]:
    models: List[Tuple[str, Forecaster, int]] = [
        (GATE, f_naive_last_week, 1),
        (f"seasonal_naive(P={lag})", f_seasonal_naive(lag), lag + 1),
    ]
    for k in ma_ks:
        models.append((f"moving_average(k={k})", f_moving_average(k), k))
    return models


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sales", required=True, help="contract sales CSV")
    ap.add_argument("--ma-k", default="2,4,8",
                    help="comma-separated moving-average windows (default 2,4,8)")
    ap.add_argument("--seasonal-lag", type=int, default=4,
                    help="seasonal-naive lag in weeks (default 4)")
    ap.add_argument("--min-history", type=int, default=0,
                    help="min history weeks per item (default: auto = max need)")
    args = ap.parse_args(argv)

    try:
        ma_ks = [int(x) for x in args.ma_k.split(",") if x.strip()]
        if not ma_ks or any(k < 1 for k in ma_ks):
            raise ValueError("--ma-k must be positive integers")
        if args.seasonal_lag < 1:
            raise ValueError("--seasonal-lag must be >= 1")
        triples = build_models(ma_ks, args.seasonal_lag)
        min_hist = args.min_history or required_history(triples)
        if min_hist < required_history(triples):
            raise ValueError(
                f"--min-history {min_hist} too small: models need "
                f"{required_history(triples)} weeks for identical folds")
        rows = load_sales(args.sales)
        weeks, series = weekly_series(rows)
        if len(weeks) <= min_hist:
            raise ValueError(
                f"only {len(weeks)} weeks of data; need more than "
                f"min-history {min_hist}")
        models = [(n, fn) for n, fn, _ in triples]
        stats, _ = evaluate(weeks, series, models, min_hist)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    text, passed = report(stats, weeks, series, min_hist, args.sales)
    print(text)
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
