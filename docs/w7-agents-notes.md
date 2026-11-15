# Week 7 notes — live-and-die agents (2026-11-13)

Plan Week 7 done-when: *watch agents die and spawn, and see whether lineages
improve.* This doc is the written observation.

## What was built

- `scripts/agents_sim.py` — stdlib-only population simulator + self-contained
  HTML replay dashboard. Each agent is a reorder policy (genome = `ma_k`,
  `safety`, same math as `reorder.py`) trading one shop's real weekly demand
  stream (the contract CSV, bucketed exactly like `backtest.py`).
- Wallet economy (SIMULATION credits, not money): revenue − order cost −
  stockout penalty − holding cost − weekly upkeep.
- Lifecycle: death at wallet ≤ 0; spawn = **parent pays** `spawn_cost` into a
  mutated child; `LINEAGE_EXTINCT` on last member death; house-money
  `IMMIGRANT` refill below `pop_min` so the watch never freezes.
- `tests/test_agents_sim.py` — 13 tests (hand-computed weeks, no-future-
  leakage twin series, determinism, spawn-funding, mutation bounds, HTML JSON
  payload parses). Full suite: **64/64 green**.

## Reproduce

```
python scripts/generate_fake_sales.py --out examples/fake_sales_52w.csv --weeks 52 --seed 42 --start 2026-01-05
python scripts/agents_sim.py --sales examples/fake_sales_52w.csv --html examples/agents_dashboard.html --log examples/agents_run.log
python -m unittest discover -s tests        # 64
```

Open `examples/agents_dashboard.html` and press Play (owner: this is the one
manual-verification item — the agent cannot see rendered HTML).

## The run (52 weeks, seed 7)

| metric | value |
|---|---|
| agents born | 189 (181 deaths, 179 spawns, 4 immigrants) |
| generations reached | 17 |
| population | cycles 6 → 24 → 8 (min 6, max 24) |
| deaths per 10-week bucket | 27 / 14 / 26 / 79 / 31 / 4 |
| survivors at week 51 | 8 (gens 0, 1, 8, 9, 10; 3 lineages) |

Deaths are spread across the whole run — no week-0 wipeout, no frozen board.

## Do lineages improve? — the answer

**On lifespan: yes, then no.** Mean lifespan by generation: founders 2.7w →
gen 3: 9.2w → gen 4: 9.8w (3.6× better), then declines for later generations
(gen 8+: ~2–3w). Selection works for the first few generations; after that
something eats the gain.

**On end wallet: descendants are worse.** gen ≥ 2 mean −45.76 vs founders
−11.27. Partly structural, not genetic: founders start with
`INITIAL_WEALTH=100`, children start with only `spawn_cost=40` (the plan's
"money coming from the parent"). Wallet therefore penalizes lineage depth by
construction; lifespan is the fairer fitness read.

**Seed check (3 seeds, 2026-11-13):** the shape repeats — lifespan rises
from founders to gen 3–4 (7/13/42: 2.7→9.2–9.8, 3.1→9.7, 3.5→14.3) then
falls; `descendants_better=False` in all three; every seed ends with
survivors (8/24/24) and cycling population 6→24. `(medium confidence)` on
the mechanism (mutational drift / 40-credit inheritance vs 100-founder
wealth / pop-cap competition) — the pattern is verified, the cause is a
hypothesis.

## Economy tuning history (why the constants are what they are)

1. `UNIT_PRICE=2.0`, no upkeep/stock: **no deaths at all** after week 0 —
   unwatchable, lineage comparison meaningless.
2. Added `UPKEEP=10`, `AGENT_INITIAL_STOCK=15` (fair cold start — kills the
   genome-independent week-0 wipeout where everyone died with no history to
   order from), harsher holding `0.35`.
3. `UNIT_PRICE=1.2` (margin 0.2/unit): deaths spread across all weeks,
   survivors persist → **current**.

Standing gate respected: no number here is claimed as a real shop's — the
demand stream is the seeded fake generator until Week 5 pilot data exists.

## Known limits

- 3 seeds checked for the lineage-shape claim (see above); the mechanism
  explanation remains a hypothesis, not a proven cause.
- `descendants_mean_wallet` compares end-of-run wallets across unequal
  lifetimes — treat the lifespan table as primary.
- Dashboard is plain HTML/JS, no framework: patterns from stock-agent (dark
  guard cards, event stream, play/pause/speed) — no code copied, per plan.
