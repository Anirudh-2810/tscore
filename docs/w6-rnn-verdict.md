# Week 6 verdict — RNN vs baselines: DROP (2026-11-13)

Plan Week 6 rule: **keep the neural model only if it clearly wins** — pooled
walk-forward WAPE strictly below both the naive gate and the best baseline.
Ties lose. This is a valid, written result (the plan says so explicitly).

## Engine substitute — flagged

The plan named `neural-engine` (`Neural net 2/neuralnet.py`) as the engine.
As of 2026-11-13 that file exists **only on the old machine** — not local, not
on GitHub. Two checks were run before substituting:

1. `Algorithm101` (https://github.com/Anirudh-2810/Algorithm101) — inspected at
   code level: **contains no trainable neural net.** Backend `server.py` is
   rule-based scoring (`eng_score`, `vel_score`, 3-window `forecast_trends`,
   composite `predict_viral_songs`); its "hidden state" line is an exponential
   moving average over genre scores; the only `LocalRNN`
   (`frontend/src/components/RnnBars.js`) is a decorative animation driven by
   `Math.sin(Date.now()/1000)` with random fixed weights. Confirms the plan's
   "NOT Algorithm101" warning.
2. `neural-engine` source unreachable → substitute: a from-scratch **Elman
   RNN** implemented in `scripts/rnn_test.py` (stdlib-only, tscore policy,
   trainable, deterministic).

If the old machine's `Neural net 2/` folder is copied over later, run it as a
cross-check — the verdict below is about data sufficiency, not the engine.

## Protocol

Same walk-forward harness as `scripts/backtest.py`: identical folds
(72 predictions/model over 14 weeks / 12 items), expanding window, per-fold
training with **target week <= test week** (unit-tested, sentinel-checked),
train-fold scaling only. Models: `naive_last_week` (gate), `seasonal_naive`,
`moving_average(k=2,4,8)`, `elman_rnn(h=8, lr=0.02)`.

## Results — all configurations lose

| seed | epochs | RNN WAPE% | vs gate 34.77% | vs best baseline (MA(8)) 28.10% |
|------|--------|-----------|----------------|-------------------------------|
| 7    | 60     | 39.92     | worse          | worse                         |
| 7    | 300    | 37.79     | worse          | worse                         |
| 1    | 60     | 39.26     | worse          | worse                         |
| 1    | 300    | 34.92     | worse (closest)| worse                         |
| 13   | 60     | 49.67     | worse          | worse                         |
| 13   | 300    | 48.82     | worse          | worse                         |
| 42   | 60     | 47.88     | worse          | worse                         |
| 42   | 300    | 47.49     | worse          | worse                         |

Default run (`--seed 7 --epochs 60`): **RNN 39.92% | gate 34.77% | MA(8) 28.10%**
→ `RNN VERDICT: DROP`, exit code 2.

## Why (honest read)

- **14 weeks of data is nowhere near enough.** Per fold the RNN trains on
  1–5 samples per item (histories must already be >= 8 weeks old). Neural nets
  are high-variance learners; at this sample size they lose to
  constant/averaging predictors, which have almost no variance to estimate.
- MA(8) wins because weekly pharmacy demand in the fake generator is
  essentially a slow-moving level + noise — averaging is the right inductive
  bias for that data-generating process.
- Seed variance (34.9–49.7%) confirms the model is estimation-noise bound,
  not under-trained: more epochs barely move it (seed 42: 47.88 → 47.49).

## Revisit condition

Re-run `python scripts/rnn_test.py --sales <real csv>` **only after the Week 5
pilot provides >= ~6 months of real weekly history** (26+ folds). Until then:
no neural model work — standing rule 1 still gates everything.

Reproduce: `python scripts/rnn_test.py --sales examples/fake_sales.csv`
(0 = KEEP, 2 = DROP, 1 = usage error). Tests: `python -m unittest discover -s tests`
(9 RNN tests incl. fold-parity vs `backtest.evaluate`, no-leakage sentinels,
determinism, verdict truth table).
