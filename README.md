# tscore

Shared forecasting core for **ShopBrain** — the offline AI back-office for small
shops (pharmacy wedge). This repo owns the data contracts, test data, and (from
Week 2) the baseline forecasts, backtest, and reorder guards that StockOffline
consumes.

Part of the ShopBrain build: [`inventory-system`](https://github.com/Anirudh-2810/inventory-system)
(StockOffline, the app shops use) imports sales in the contract defined here.

## Status

- [x] **Week 1 — data contract v1** — sales CSV + forecast output CSV ([`docs/data-contract.md`](docs/data-contract.md))
- [x] **Week 1 — fake sales generator** — weekly + seasonal patterns, seeded, zero-dependency
- [x] **Week 2 — baselines + walk-forward backtest** ([`scripts/backtest.py`](scripts/backtest.py))
      — on `examples/fake_sales.csv` (14 weeks, 12 items, 72 folds/model):
      **GATE PASSED**, `moving_average(k=8)` WAPE 28.10% vs naive gate 34.77%
      (seasonal_naive P=4 33.48%, ma(k=4) 28.28% also beat the gate)
- [x] **Week 3 — reorder math + guards** ([`scripts/reorder.py`](scripts/reorder.py))
      — lead-time × safety-buffer need, max-order cap, N× last-order cap,
      weekly budget with urgent-first allocation; grid-tested invariants
- [x] **Week 4 — weekly report (tscore side)** ([`scripts/report.py`](scripts/report.py))
      — owner-readable `report.txt` (order list, running-out, dead stock)
      + contract `forecast.csv`; confidence is measured backtest WAPE
      (0.5 placeholder only while history is too short to backtest — disclosed)
- [ ] Week 4 (StockOffline side): Forecast screen with approve/reject
- [ ] Week 5+: real pilot (blocked on shop permission — owner task)

## Quick start

```bash
# generate 13 weeks of fake pharmacy sales (seeded, reproducible)
python scripts/generate_fake_sales.py --out examples/fake_sales.csv

# run all tests (contract + backtest + reorder + report)
python -m unittest discover -s tests -v

# walk-forward backtest with the gate verdict (exit 0 = gate passed, 2 = failed)
python scripts/backtest.py --sales examples/fake_sales.csv

# reorder math demo (guards: max order, N x last order, budget)
python scripts/reorder.py --demo

# weekly report: sales + stock in -> report.txt + forecast.csv out
python scripts/report.py --sales examples/fake_sales.csv \
  --stock examples/stock_example.csv --out-dir out --budget 5000
```

## Data contracts (summary — full spec in docs/data-contract.md)

**Sales CSV** (input, produced by shops or the generator):

```csv
item,date,quantity,price
PARACETAMOL-500,2026-10-01,3,30.00
```

**Forecast output CSV** (produced by tscore, consumed by StockOffline's report):

```csv
item,predicted_demand,suggested_reorder,confidence
PARACETAMOL-500,42,50,0.73
```

## Standing gates (from the ShopBrain plan)

1. A baseline must beat naive "reorder what sold last week" before any model
   work happens.
2. Real shop data stays local and out of git — only generated/hand-written
   samples live in `examples/`.
3. No new project work until a shop has paid.

## Related

- ShopBrain plan: `wiki/01-Areas/Business/plans/shopbrain-agency-launch.md` in
  the [Second-Brain](https://github.com/Anirudh-2810/Second-Brain) vault
- StockOffline: [inventory-system](https://github.com/Anirudh-2810/inventory-system)
