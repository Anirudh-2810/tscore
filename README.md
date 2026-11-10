# tscore

Shared forecasting core for **ShopBrain** — the offline AI back-office for small
shops (pharmacy wedge). This repo owns the data contracts, test data, and (from
Week 2) the baseline forecasts, backtest, and reorder guards that StockOffline
consumes.

Part of the ShopBrain build: [`inventory-system`](https://github.com/Anirudh-2810/inventory-system)
(StockOffline, the app shops use) imports sales in the contract defined here.

## Status (Week 1)

- [x] **Data contract v1** — sales CSV + forecast output CSV ([`docs/data-contract.md`](docs/data-contract.md))
- [x] **Fake sales generator** — weekly + seasonal patterns, seeded, zero-dependency
- [x] Contract tests (structure, determinism, date sanity)
- [ ] Week 2: moving-average + seasonal-naive baselines, walk-forward backtest
- [ ] Week 3: reorder math + guards (max order, budget cap, last-order multiple)
- [ ] Week 4: weekly report format

## Quick start

```bash
# generate 13 weeks of fake pharmacy sales (seeded, reproducible)
python scripts/generate_fake_sales.py --out examples/fake_sales.csv

# run contract tests
python -m unittest discover -s tests -v
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
