"""Generate fake pharmacy sales CSVs for ShopBrain testing (contract v1).

Zero-dependency (stdlib only), seeded and reproducible. Patterns baked in:
  * weekly rhythm   — weekend uplift + pay-day (1st-5th) bump
  * seasonal curve  — smooth yearly sinusoid with a flu-season bump
  * noise           — Poisson day-to-day variation, occasional zero days

Usage:
    python scripts/generate_fake_sales.py --out examples/fake_sales.csv
    python scripts/generate_fake_sales.py --items 6 --weeks 8 --seed 7 --out /tmp/s.csv
"""

import argparse
import csv
import math
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# (item, base daily rate, unit price) — a small generic pharmacy shelf
DEFAULT_CATALOG = [
    ("PARACETAMOL-500", 6.0, 30.00),
    ("IBUPROFEN-400",   3.5, 45.00),
    ("CETIRIZINE-10",   2.8, 40.00),
    ("OMEPRAZOLE-20",   2.2, 60.00),
    ("AZITHROMYCIN-500",1.4, 120.00),
    ("ORS-SACHET",      4.5, 20.00),
    ("GLUCOSE-STRIP",   1.8, 90.00),
    ("BP-MONITOR-CUFF", 0.35, 850.00),
    ("VITAMIN-C-1000",  2.5, 55.00),
    ("COUGH-SYRUP-100", 2.0, 75.00),
    ("BANDAGE-ROLL",    1.6, 25.00),
    ("INSULIN-PEN-ND",  0.6, 650.00),
]

WEEKDAY_FACTOR = [1.05, 1.0, 1.0, 1.0, 1.05, 1.1, 1.15]  # Mon..Sun
SEASONAL_PEAK_DAY = 200   # ~mid-July: monsoon/flu-season bump
SEASONAL_AMPLITUDE = 0.30
FLU_WINDOW = (300, 365)   # Nov-Dec extra bump


def seasonal_factor(doy: int) -> float:
    wave = 1.0 + SEASONAL_AMPLITUDE * math.sin(2 * math.pi * (doy - SEASONAL_PEAK_DAY) / 365.0)
    flu = 0.25 if FLU_WINDOW[0] <= doy <= FLU_WINDOW[1] else 0.0
    return wave + flu


def pay_day_factor(day: int) -> float:
    return 1.35 if day <= 5 else 1.0


def poisson(rng: random.Random, lam: float) -> int:
    """Knuth's Poisson sampler — fine for the small lambdas used here."""
    if lam <= 0:
        return 0
    limit = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def generate(catalog, weeks: int, start: date, seed: int, trend_per_week: float = 0.005):
    """Yield contract-valid sale rows (item, date, quantity, price)."""
    rng = random.Random(seed)
    rows = []
    total_days = weeks * 7
    for offset in range(total_days):
        d = start + timedelta(days=offset)
        week_no = offset // 7
        trend = 1.0 + trend_per_week * week_no
        for item, base, price in catalog:
            lam = (base
                   * WEEKDAY_FACTOR[d.weekday()]
                   * seasonal_factor(d.timetuple().tm_yday)
                   * pay_day_factor(d.day)
                   * trend)
            qty = poisson(rng, lam)
            if qty > 0:
                rows.append((item, d.isoformat(), qty, f"{price:.2f}"))
    rows.sort(key=lambda r: (r[1], r[0]))
    return rows


def write_csv(rows, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "date", "quantity", "price"])
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate fake sales CSV (ShopBrain contract v1)")
    ap.add_argument("--out", default="examples/fake_sales.csv", help="output CSV path")
    ap.add_argument("--weeks", type=int, default=13, help="weeks of history (default 13)")
    ap.add_argument("--items", type=int, default=len(DEFAULT_CATALOG),
                    help="how many catalog items to use (default: all)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    ap.add_argument("--start", default=None, help="first date YYYY-MM-DD (default: today - weeks*7)")
    args = ap.parse_args(argv)

    if args.weeks < 1 or args.items < 1:
        print("--weeks and --items must be >= 1", file=sys.stderr)
        return 2
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start = date.today() - timedelta(days=args.weeks * 7)

    catalog = DEFAULT_CATALOG[: min(args.items, len(DEFAULT_CATALOG))]
    rows = generate(catalog, args.weeks, start, args.seed)
    out = Path(args.out)
    write_csv(rows, out)
    print(f"wrote {len(rows)} sales rows for {len(catalog)} items "
          f"({args.weeks} weeks from {start.isoformat()}, seed={args.seed}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
