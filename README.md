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
- [x] **Week 6 — neural test: DROP** ([`scripts/rnn_test.py`](scripts/rnn_test.py),
      [`docs/w6-rnn-verdict.md`](docs/w6-rnn-verdict.md))
      — from-scratch Elman RNN (engine substitute: `neural-engine` source lives
      only on the old machine; `Algorithm101` inspected — no trainable NN,
      EMA + decorative viz). Walk-forward WAPE **39.92%** vs gate 34.77% vs
      MA(8) 28.10% (defaults); loses under all 8 seed×epoch configs tried
      (34.92–49.67%) → **DROP, revisit only with >=6 months of pilot data**
- [x] **Week 7 — live-and-die agents + replay dashboard** ([`scripts/agents_sim.py`](scripts/agents_sim.py),
      [`docs/w7-agents-notes.md`](docs/w7-agents-notes.md))
      — population of MA(k)×safety reorder agents on one shop's demand stream,
      wallet economy (revenue − orders − stockouts − holding − upkeep), death
      at 0, parent-funded mutated spawns, immigrant refill. 52w/seed-7 run:
      189 agents, 17 generations, deaths in every week-bucket, 8 survivors.
      Lineage answer: lifespan peaks at gen 3–4 (2.7w → 9.8w) then declines;
      descendant wallets worse (structural: children inherit 40, founders 100)
      — shape verified on 3 seeds. Self-contained `agents_dashboard.html` replay
- [x] **Week 8 (agent part) — demo driver + case-study skeleton** ([`scripts/demo.py`](scripts/demo.py),
      [case-study draft](https://github.com/Anirudh-2810/inventory-system/blob/main/marketing/case-study.md))
      — `python scripts/demo.py` runs the whole pipeline end to end (fake data →
      gate → report → agent replay) with fixed seeds so it is byte-stable any
      day; case study is a `(TBC)` skeleton — no numbers before the pilot
- [x] Week 4 (StockOffline side): Forecast screen with approve/reject (app repo `b8f08b3`)
- [ ] Week 5+: real pilot (blocked on shop permission — owner task)
- [ ] Week 8 (owner part): approach 10 shops, let their answers set the price

## Quick start

```bash
# generate 13 weeks of fake pharmacy sales (seeded, reproducible)
python scripts/generate_fake_sales.py --out examples/fake_sales.csv

# run all tests (contract + backtest + reorder + report + rnn + agents)
python -m unittest discover -s tests -v

# walk-forward backtest with the gate verdict (exit 0 = gate passed, 2 = failed)
python scripts/backtest.py --sales examples/fake_sales.csv

# Week 6 neural test (exit 0 = KEEP, 2 = DROP; currently DROP — see docs/w6-rnn-verdict.md)
python scripts/rnn_test.py --sales examples/fake_sales.csv

# Week 7 live-and-die agents: 52-week run + replay dashboard (open the HTML, press Play)
python scripts/generate_fake_sales.py --out examples/fake_sales_52w.csv --weeks 52 --seed 42 --start 2026-01-05
python scripts/agents_sim.py --sales examples/fake_sales_52w.csv \
  --html examples/agents_dashboard.html --log examples/agents_run.log

# Week 8 one-command demo: the whole pipeline end to end (fake data -> gate ->
# report -> agent replay). Use --dry-run to list the steps, --with-rnn to add
# the internal RNN test (expected exit 2 = DROP).
python scripts/demo.py

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

---

Proprietary — © StockOffline. All rights reserved.
