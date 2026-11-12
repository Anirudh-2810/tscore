#!/usr/bin/env python3
"""Weekly reorder report for shop owners (plan Week 4, tscore side).

One command turns a sales CSV + current stock into:
  report.txt    - the owner-facing weekly report: order list, running-out
                  alerts, dead stock, short summary (readable without help)
  forecast.csv  - data-contract v1 forecast (item,predicted_demand,
                  suggested_reorder,confidence) for StockOffline to import

Confidence is MEASURED, not guessed: per item it is 1 - WAPE from the same
walk-forward backtest that gates the model (items too short for folds fall
back to the pooled WAPE, flagged in the report).

Usage:
  python scripts/report.py --sales sales.csv --stock stock.csv --out-dir out
  stock CSV columns: item,on_hand,unit_price[,last_order_qty]

Exit codes: 0 ok, 1 input/usage error.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest as bt        # noqa: E402
import reorder as ro         # noqa: E402

STOCK_COLUMNS_REQUIRED = ("item", "on_hand", "unit_price")
FORECAST_HEADER = ("item", "predicted_demand", "suggested_reorder", "confidence")


def load_stock(path: str) -> Dict[str, Tuple[int, float, Optional[int]]]:
    """item -> (on_hand, unit_price, last_order_qty|None)."""
    stock: Dict[str, Tuple[int, float, Optional[int]]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [(h or "").strip() for h in (reader.fieldnames or [])]
        missing = [c for c in STOCK_COLUMNS_REQUIRED if c not in header]
        if missing:
            raise ValueError(
                f"not a stock CSV: missing column(s) {', '.join(missing)}")
        key_map = {(raw or "").strip(): raw for raw in (reader.fieldnames or [])}
        for raw in reader:
            row = {name: raw.get(r) for name, r in key_map.items()}
            item = (row.get("item") or "").strip()
            if not item:
                continue
            try:
                on_hand = int(float(row.get("on_hand") or 0))
                price = float(row.get("unit_price") or 0)
                last_raw = (row.get("last_order_qty") or "").strip()
                last = int(float(last_raw)) if last_raw else None
            except (ValueError, TypeError):
                raise ValueError(f"bad stock row for {item!r}")
            stock[item] = (max(0, on_hand), max(0.0, price), last)
    if not stock:
        raise ValueError("no usable rows in stock CSV")
    return stock


def per_item_wape(preds: Sequence[bt.Pred]) -> Dict[str, float]:
    """item -> WAPE over its walk-forward predictions."""
    agg: Dict[str, Tuple[float, float]] = {}
    for item, _, p, a in preds:
        ae, act = agg.get(item, (0.0, 0.0))
        agg[item] = (ae + abs(p - a), act + a)
    return {i: (ae / act if act > 0 else 0.0) for i, (ae, act) in agg.items()}


def build_report(
    sales_path: str,
    stock: Dict[str, Tuple[int, float, Optional[int]]],
    model_name: str = "moving_average(k=8)",
    alert_cover: float = 2.0,
    dead_weeks: int = 4,
    guards: Optional[ro.Guards] = None,
) -> Tuple[str, List[Tuple[str, float, int, float]]]:
    """Returns (report_text, forecast_rows) where forecast_rows are
    (item, predicted_demand, suggested_reorder, confidence)."""
    guards = guards or ro.Guards()
    rows = bt.load_sales(sales_path)
    weeks, series = bt.weekly_series(rows)
    triples = bt.build_models([8], 4)
    by_name = {n: fn for n, fn, _ in triples}
    if model_name not in by_name:
        raise ValueError(f"unknown model {model_name!r}; have: "
                         f"{', '.join(by_name)}")
    min_hist = bt.required_history(triples)
    models = [(n, fn) for n, fn, _ in triples]
    stats, preds = bt.evaluate(weeks, series, models, min_hist)
    gate_pass = bt.verdict(stats)[2]
    n_folds = int(stats[bt.GATE]["n"])
    pooled_wape = bt.wape(stats[model_name]) or 0.0
    item_wape = per_item_wape(preds)
    fn = by_name[model_name]

    # -- forecasts ---------------------------------------------------------
    forecast: Dict[str, float] = {}
    confidence: Dict[str, float] = {}
    short_hist: List[str] = []
    for item, vals in series.items():
        hist = [v for v in vals if v is not None]
        if not hist:
            continue
        if len(hist) >= min_hist:
            forecast[item] = fn(hist)
            confidence[item] = 0.5 if n_folds == 0 else round(
                max(0.0, min(1.0, 1.0 - item_wape.get(item, pooled_wape))), 3)
        else:
            forecast[item] = bt.f_naive_last_week(hist)   # documented fallback
            confidence[item] = 0.5 if n_folds == 0 else round(
                max(0.0, min(1.0, 1.0 - pooled_wape)), 3)
            short_hist.append(item)

    # -- stock-aware reorder lines ----------------------------------------
    warnings: List[str] = []
    lines: List[ro.Line] = []
    for item, fc in forecast.items():
        if item in stock:
            on_hand, price, last = stock[item]
        else:
            on_hand, price, last = 0, 0.0, None
            warnings.append(f"{item}: no stock record — assumed on hand 0, price 0")
        lines.append(ro.Line(item, forecast=fc, on_hand=on_hand,
                             unit_price=price, last_order_qty=last))
    results = ro.allocate(lines, guards)
    by_item = {r.item: r for r in results}

    # -- alert buckets -----------------------------------------------------
    running_out: List[Tuple[str, float]] = []
    dead: List[Tuple[str, int]] = []
    wanted_but_blocked: List[Tuple[str, int, List[str]]] = []
    last_n = weeks[-dead_weeks:] if dead_weeks > 0 else weeks
    last_idx = {weeks.index(w) for w in last_n}
    for line in lines:
        cover = ro.weeks_of_cover(line)
        # out-of-stock items (cover 0) are the MOST urgent — include them
        if line.forecast > 0 and cover < alert_cover:
            running_out.append((line.item, cover))
        res = by_item[line.item]
        raw = ro.raw_need(line.forecast, line.on_hand, guards)
        if raw > 0 and res.qty == 0:
            wanted_but_blocked.append(
                (line.item, raw, res.capped_by or ["unknown guard"]))
        vals = series[line.item]
        recent = sum(v for i, v in enumerate(vals)
                     if v is not None and i in last_idx)
        if line.on_hand > 0 and recent == 0:
            dead.append((line.item, line.on_hand))

    # -- assemble ----------------------------------------------------------
    ordered = [r for r in results if r.qty > 0]
    total_spend = sum(r.spend for r in ordered)
    out: List[str] = []
    out.append(f"ShopBrain weekly reorder report - generated {date.today().isoformat()}")
    out.append(f"Data: {Path(sales_path).name} ({len(weeks)} weeks, "
               f"{len(series)} items), stock records: {len(stock)}")
    gate_wape = bt.wape(stats[bt.GATE])
    if n_folds == 0:
        out.append(f"Model: {model_name} - no backtest folds yet "
                   f"(need more than {min_hist} weeks of history) - "
                   "GATE: not evaluable")
    else:
        gate_w = gate_wape or 0.0
        if model_name == bt.GATE:
            label = "gate baseline mode"
        elif gate_pass:
            label = "GATE PASSED"
        else:
            others = [bt.wape(s) for n, s in stats.items()
                      if n != bt.GATE and s["n"] > 0]
            tie = any(w is not None and abs(w - gate_w) < 1e-12
                      for w in others)
            label = ("GATE TIED - best baseline only matches naive"
                     " (rule 1 needs a BEAT)" if tie
                     else "GATE FAILED (see backtest)")
        out.append(f"Model: {model_name} - backtest WAPE "
                   f"{100 * pooled_wape:.1f}% vs naive gate "
                   f"{100 * gate_w:.1f}% - {label}")
    out.append("")
    budget_txt = (f" / budget Rs {guards.budget_cap:.0f}"
                  if guards.budget_cap is not None else "")
    out.append(f"ORDER THIS WEEK ({len(ordered)} items, "
               f"total Rs {total_spend:.2f}{budget_txt})")
    if ordered:
        for r in sorted(ordered, key=lambda r: r.item):
            line = by_item[r.item]
            caps = f"  [{', '.join(r.capped_by)}]" if r.capped_by else ""
            out.append(f"  {r.item:<22} order {r.qty:>5}"
                       f" (forecast {r.forecast:.0f}, on hand {line.on_hand})"
                       f"  Rs {r.spend:.2f}{caps}")
    else:
        out.append("  (nothing to order)")
    out.append("")
    out.append(f"WANTED BUT NOT ORDERED ({len(wanted_but_blocked)} items) - "
               + ("demand exists but a limit blocked it"
                  if wanted_but_blocked else "none"))
    for item, raw, caps in sorted(wanted_but_blocked):
        out.append(f"  {item:<22} needed {raw:>4} - blocked by: "
                   f"{', '.join(caps)}")
    out.append("")
    out.append(f"RUNNING OUT (cover < {alert_cover:g} weeks): "
               + (f"{len(running_out)} items" if running_out else "none"))
    for item, cover in sorted(running_out, key=lambda x: x[1]):
        out.append(f"  {item:<22} {cover:.1f} weeks cover")
    out.append("")
    out.append(f"DEAD STOCK (no sales in {dead_weeks} weeks, still on hand): "
               + (f"{len(dead)} items" if dead else "none"))
    for item, qty in sorted(dead):
        out.append(f"  {item:<22} {qty} units stuck")
    not_ordered = [r for r in results if r.qty == 0]
    out.append("")
    out.append(f"NOT ORDERED (enough stock): {len(not_ordered)} items")
    if short_hist:
        out.append("")
        out.append("NOTES")
        out.append(f"  - short history (used last-week naive): "
                   f"{', '.join(sorted(short_hist))}")
    if n_folds == 0:
        if not short_hist:
            out.append("")
            out.append("NOTES")
        out.append("  - confidence = 0.5 placeholder for all items "
                   "(no backtest folds yet)")
    if warnings:
        out.append("")
        out.append("WARNINGS")
        for w in warnings:
            out.append(f"  - {w}")
    forecast_rows = [
        (r.item, round(forecast[r.item], 2), r.qty, confidence[r.item])
        for r in results
    ]
    return "\n".join(out), forecast_rows


def write_forecast_csv(path: Path, forecast_rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FORECAST_HEADER)
        for item, pred, qty, conf in forecast_rows:
            w.writerow([item, pred, qty, conf])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sales", required=True, help="contract sales CSV")
    ap.add_argument("--stock", required=True,
                    help="stock CSV: item,on_hand,unit_price[,last_order_qty]")
    ap.add_argument("--out-dir", required=True, help="where report.txt + forecast.csv go")
    ap.add_argument("--model", default="moving_average(k=8)",
                    help="forecaster name (default: the gate-tested winner)")
    ap.add_argument("--budget", type=float, default=None,
                    help="weekly spend cap (Rs)")
    ap.add_argument("--safety", type=float, default=0.20,
                    help="safety buffer fraction (default 0.20)")
    ap.add_argument("--lead-time", type=int, default=1,
                    help="delivery weeks to cover (default 1)")
    ap.add_argument("--max-order", type=int, default=None,
                    help="per-line quantity cap")
    ap.add_argument("--max-multiple", type=float, default=None,
                    help="cap = this x last order qty")
    ap.add_argument("--alert-cover", type=float, default=2.0,
                    help="running-out threshold in weeks (default 2)")
    ap.add_argument("--dead-weeks", type=int, default=4,
                    help="no-sales window for dead stock (default 4)")
    args = ap.parse_args(argv)

    try:
        stock = load_stock(args.stock)
        guards = ro.Guards(
            safety_buffer=args.safety, lead_time_weeks=args.lead_time,
            max_order_qty=args.max_order, max_multiple_of_last=args.max_multiple,
            budget_cap=args.budget)
        text, rows = build_report(args.sales, stock, model_name=args.model,
                                  alert_cover=args.alert_cover,
                                  dead_weeks=args.dead_weeks, guards=guards)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
        write_forecast_csv(out_dir / "forecast.csv", rows)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(text)
    print(f"\nWrote {out_dir / 'report.txt'} and {out_dir / 'forecast.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
