#!/usr/bin/env python3
"""Week 6 neural test: can a from-scratch RNN beat the ShopBrain baselines?

Plan flag (2026-11-13): the plan names `neural-engine` (Neural net 2/
neuralnet.py) as the engine, but that file exists only on the old machine -
not local, not on GitHub - so the RNN is implemented here, stdlib-only per
tscore policy. Same walk-forward protocol and no-leakage rule as
scripts/backtest.py: for every test week t the RNN trains only on samples
whose target week is <= t.

Models on identical folds: naive_last_week (gate), seasonal_naive(P),
moving_average(k=2,4,8), elman_rnn(h).

Keep rule (plan Week 6): KEEP only if the RNN's pooled WAPE is strictly below
BOTH the gate and the best baseline; ties lose. DROP is a valid result and is
written up.

Exit codes: 0 = KEEP, 2 = DROP, 1 = input/usage error.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import backtest as bt
except ImportError:                      # run as scripts/rnn_test.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import backtest as bt

Fold = Tuple[str, int, List[float], float]   # (item, cut, hist, actual)
Seq = Tuple[List[float], List[float]]        # (inputs, targets), scaled


# -- data plumbing (mirrors backtest.evaluate's fold rule) ----------------
def iter_folds(series: Dict[str, List[Optional[float]]],
               weeks: Sequence[date], min_hist: int) -> Iterable[Fold]:
    """The exact folds backtest.evaluate scores: per item, every cut with
    hist_len >= min_hist whose next week exists. (A test asserts parity.)"""
    for item, vals in series.items():
        first = next((i for i, v in enumerate(vals) if v is not None), None)
        if first is None:
            continue
        for cut in range(first, len(weeks) - 1):
            if cut - first + 1 < min_hist:
                continue
            actual = vals[cut + 1]
            if actual is None:
                continue
            hist = [v for v in vals[first:cut + 1] if v is not None]
            yield item, cut, hist, float(actual)


def train_samples(series: Dict[str, List[Optional[float]]], cut: int,
                  min_hist: int) -> List[Tuple[List[float], float]]:
    """Training set available when predicting week `cut`: every (hist, target)
    whose target week index is <= cut. Nothing from the future - unit-tested
    with sentinel values in tests/test_rnn_test.py.
    """
    out: List[Tuple[List[float], float]] = []
    for vals in series.values():
        first = next((i for i, v in enumerate(vals) if v is not None), None)
        if first is None:
            continue
        for c in range(first, cut):          # target week = c+1 <= cut
            if c - first + 1 < min_hist:
                continue
            target = vals[c + 1]
            if target is None:
                continue
            hist = [v for v in vals[first:c + 1] if v is not None]
            out.append((hist, float(target)))
    return out


def scale_at(series: Dict[str, List[Optional[float]]], cut: int,
             memo: Dict[int, float]) -> float:
    """Max observed value in weeks 0..cut (train-fold stats only: no leak)."""
    if cut in memo:
        return memo[cut]
    mx = 0.0
    for vals in series.values():
        for i, v in enumerate(vals):
            if i > cut:
                break
            if v is not None and v > mx:
                mx = v
    memo[cut] = mx if mx > 0 else 1.0
    return memo[cut]


def to_seq(hist: Sequence[float], target: float, scale: float) -> Seq:
    """One training sequence: after each history value predict the next one,
    last step predicts the sample's target."""
    xs = [v / scale for v in hist]
    tars = [v / scale for v in hist[1:]] + [target / scale]
    return xs, tars


# -- the RNN --------------------------------------------------------------
class ElmanRNN:
    """One hidden layer Elman RNN: scalar weekly input, tanh hidden, linear
    output. Trained by full backpropagation-through-time over each sample's
    history (a few steps - the weekly series is short), per-sample SGD with
    per-component gradient clipping. Stdlib only, deterministic given a seed.
    """

    def __init__(self, hidden: int, seed: int, lr: float,
                 clip: float = 5.0) -> None:
        rng = random.Random(seed)
        self.hidden = hidden
        self.lr = lr
        self.clip = clip
        s_hid = 1.0 / math.sqrt(hidden)
        self.wx = [rng.gauss(0.0, 1.0) for _ in range(hidden)]
        self.wh = [[rng.gauss(0.0, s_hid) for _ in range(hidden)]
                   for _ in range(hidden)]
        self.bh = [0.0] * hidden
        self.wo = [rng.gauss(0.0, s_hid) for _ in range(hidden)]
        self.bo = 0.0

    def clone(self) -> "ElmanRNN":
        m = ElmanRNN.__new__(ElmanRNN)
        m.hidden = self.hidden
        m.lr = self.lr
        m.clip = self.clip
        m.wx = list(self.wx)
        m.wh = [row[:] for row in self.wh]
        m.bh = list(self.bh)
        m.wo = list(self.wo)
        m.bo = self.bo
        return m

    # -- forward --------------------------------------------------------
    def _forward(self, xs: Sequence[float]) -> Tuple[List[List[float]], List[float]]:
        h = [0.0] * self.hidden              # h at t=-1: zeros
        hs = [h]
        ys: List[float] = []
        for x in xs:
            prev = h
            h = []
            for j in range(self.hidden):
                pre = self.wx[j] * x + self.bh[j]
                row = self.wh[j]
                acc = 0.0
                for k in range(self.hidden):
                    acc += row[k] * prev[k]
                h.append(math.tanh(pre + acc))
            hs.append(h)
            y = self.bo
            for j in range(self.hidden):
                y += self.wo[j] * h[j]
            ys.append(y)
        return hs, ys

    def predict(self, xs: Sequence[float]) -> float:
        _, ys = self._forward(xs)
        return ys[-1] if ys else 0.0

    # -- training -------------------------------------------------------
    def _train_sample(self, xs: List[float], tars: List[float]) -> float:
        n = len(xs)
        hs, ys = self._forward(xs)
        loss = sum((ys[i] - tars[i]) ** 2 for i in range(n)) / n

        h = self.hidden
        gwx = [0.0] * h
        gwh = [[0.0] * h for _ in range(h)]
        gbh = [0.0] * h
        gwo = [0.0] * h
        gbo = 0.0
        dpre_next: Optional[List[float]] = None
        for i in range(n - 1, -1, -1):
            hi = hs[i + 1]                   # hidden after step i
            hprev = hs[i]                    # hidden before step i
            dy = 2.0 * (ys[i] - tars[i]) / n
            dh = [dy * self.wo[j] for j in range(h)]
            if dpre_next is not None:
                for j in range(h):
                    row = self.wh[j]
                    acc = 0.0
                    for k in range(h):
                        acc += row[k] * dpre_next[k]
                    dh[j] += acc
            dpre = [dh[j] * (1.0 - hi[j] * hi[j]) for j in range(h)]
            gbo += dy
            x_i = xs[i]
            for j in range(h):
                gwo[j] += dy * hi[j]
                gbh[j] += dpre[j]
                gwx[j] += dpre[j] * x_i
                grow = gwh[j]
                hp = hprev
                for k in range(h):
                    grow[k] += dpre[j] * hp[k]
            dpre_next = dpre

        def clip(v: float) -> float:
            return max(-self.clip, min(self.clip, v))

        self.bo -= self.lr * clip(gbo)
        for j in range(h):
            self.wo[j] -= self.lr * clip(gwo[j])
            self.bh[j] -= self.lr * clip(gbh[j])
            self.wx[j] -= self.lr * clip(gwx[j])
            wrow = self.wh[j]
            grow = gwh[j]
            for k in range(h):
                wrow[k] -= self.lr * clip(grow[k])
        return loss

    def fit(self, samples: Sequence[Seq], epochs: int, seed: int) -> None:
        if not samples:
            return
        rng = random.Random(seed)
        order = list(range(len(samples)))
        for _ in range(epochs):
            rng.shuffle(order)
            for idx in order:
                self._train_sample(*samples[idx])


# -- training snapshots per test week -------------------------------------
def train_snapshots(series: Dict[str, List[Optional[float]]], cuts: Iterable[int],
                    min_hist: int, epochs: int, hidden: int, lr: float,
                    seed: int) -> Dict[int, ElmanRNN]:
    """One model per test week, warm-started in ascending cut order.

    Training data only grows with the cut (never sees week > cut), and each
    snapshot is cloned at its cut so later training cannot rewrite it.
    """
    snaps: Dict[int, ElmanRNN] = {}
    scale_memo: Dict[int, float] = {}
    model: Optional[ElmanRNN] = None
    for cut in sorted(set(cuts)):
        scale = scale_at(series, cut, scale_memo)
        samples = [to_seq(h, t, scale)
                   for h, t in train_samples(series, cut, min_hist)]
        if model is None:
            model = ElmanRNN(hidden, seed, lr)
        if samples:
            model.fit(samples, epochs, seed + cut)
        snaps[cut] = model.clone()
    if not snaps:                            # no folds at all: still return one
        snaps[0] = ElmanRNN(hidden, seed, lr)
    return snaps


def rnn_predict(snap: ElmanRNN, hist: Sequence[float], scale: float) -> float:
    """Scaled forward pass, un-scaled, clamped at 0 (no negative demand)."""
    xs = [v / scale for v in hist]
    return max(0.0, snap.predict(xs) * scale)


# -- evaluation -----------------------------------------------------------
def evaluate_rnn(folds: Sequence[Fold], snaps: Dict[int, ElmanRNN],
                 series: Dict[str, List[Optional[float]]]) -> Tuple[Dict[str, float], List[Tuple[str, int, float, float]]]:
    """Score the RNN on the same folds as backtest.evaluate (identical
    stats dict shape + the per-item carry-over stock simulation)."""
    stats = {"ae": 0.0, "se": 0.0, "act": 0.0, "n": 0.0, "under": 0.0,
             "stockouts": 0.0}
    preds: List[Tuple[str, int, float, float]] = []
    scale_memo: Dict[int, float] = {}
    first_seen: Dict[str, float] = {}
    pairs: Dict[str, List[Tuple[float, float]]] = {}
    for item, cut, hist, actual in folds:
        scale = scale_at(series, cut, scale_memo)
        p = rnn_predict(snaps[cut], hist, scale)
        preds.append((item, cut + 1, p, actual))
        stats["ae"] += abs(p - actual)
        stats["se"] += (p - actual) ** 2
        stats["act"] += actual
        stats["n"] += 1
        if p < actual:
            stats["under"] += 1
        if item not in first_seen:
            first_seen[item] = hist[-1]      # start: last week's sales on hand
        pairs.setdefault(item, []).append((p, actual))
    for item, ps in pairs.items():
        stats["stockouts"] += bt.run_stock(ps, first_seen[item])
    return stats, preds


# -- verdict --------------------------------------------------------------
def keep_decision(rnn_w: Optional[float], gate_w: Optional[float],
                  base_w: Optional[float]) -> bool:
    """Plan Week 6 rule: KEEP only if strictly below BOTH - ties lose."""
    if rnn_w is None or gate_w is None or base_w is None:
        return False
    return rnn_w < gate_w and rnn_w < base_w


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{100 * v:.2f}%"


# -- main -----------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sales", required=True, help="contract sales CSV")
    ap.add_argument("--epochs", type=int, default=60,
                    help="training epochs per test week (default 60)")
    ap.add_argument("--hidden", type=int, default=8,
                    help="RNN hidden units (default 8)")
    ap.add_argument("--lr", type=float, default=0.02,
                    help="SGD learning rate (default 0.02)")
    ap.add_argument("--seed", type=int, default=7,
                    help="RNG seed for init + shuffle (default 7)")
    ap.add_argument("--ma-k", default="2,4,8",
                    help="comma-separated moving-average windows (default 2,4,8)")
    ap.add_argument("--seasonal-lag", type=int, default=4,
                    help="seasonal-naive lag in weeks (default 4)")
    ap.add_argument("--min-history", type=int, default=0,
                    help="min history weeks per item (default: auto)")
    args = ap.parse_args(argv)

    try:
        if args.epochs < 1:
            raise ValueError("--epochs must be >= 1")
        if args.hidden < 1:
            raise ValueError("--hidden must be >= 1")
        if args.lr <= 0:
            raise ValueError("--lr must be > 0")
        ma_ks = [int(x) for x in args.ma_k.split(",") if x.strip()]
        if not ma_ks or any(k < 1 for k in ma_ks):
            raise ValueError("--ma-k must be positive integers")
        if args.seasonal_lag < 1:
            raise ValueError("--seasonal-lag must be >= 1")
        triples = bt.build_models(ma_ks, args.seasonal_lag)
        min_hist = args.min_history or bt.required_history(triples)
        if min_hist < bt.required_history(triples):
            raise ValueError(
                f"--min-history {min_hist} too small: models need "
                f"{bt.required_history(triples)} weeks for identical folds")
        rows = bt.load_sales(args.sales)
        weeks, series = bt.weekly_series(rows)
        if len(weeks) <= min_hist:
            raise ValueError(
                f"only {len(weeks)} weeks of data; need more than "
                f"min-history {min_hist} weeks")

        base_models = [(n, fn) for n, fn, _ in triples]
        base_stats, base_preds = bt.evaluate(weeks, series, base_models,
                                             min_hist)
        folds = list(iter_folds(series, weeks, min_hist))

        # Parity guard: our folds must be exactly the backtest's folds.
        # (iter_folds yields cut; evaluate's preds carry cut+1 = target week.)
        mine = {(i, t + 1) for i, t, _, _ in folds}
        theirs = {(i, t) for i, t, _, _ in base_preds}
        if mine != theirs:
            raise RuntimeError(
                f"fold mismatch vs backtest.evaluate: "
                f"only-mine={sorted(mine - theirs)[:3]} "
                f"only-theirs={sorted(theirs - mine)[:3]}")

        snaps = train_snapshots(series, (c for _, c, _, _ in folds), min_hist,
                                args.epochs, args.hidden, args.lr, args.seed)
        rnn_stats, _ = evaluate_rnn(folds, snaps, series)
    except (ValueError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    key = f"elman_rnn(h={args.hidden})"
    combined: Dict[str, Dict[str, float]] = dict(base_stats)
    combined[key] = rnn_stats
    text, _ = bt.report(combined, weeks, series, min_hist, args.sales)
    print(text)

    rnn_w = bt.wape(rnn_stats)
    gate_w = bt.wape(base_stats[bt.GATE])
    base_ws = {n: bt.wape(s) for n, s in base_stats.items()
               if n != bt.GATE and s["n"] > 0}
    base_ws = {n: v for n, v in base_ws.items() if v is not None}
    best_n = min(base_ws, key=base_ws.get) if base_ws else None
    best_w = base_ws[best_n] if best_n else None

    print(f"RNN experiment ({key}, epochs={args.epochs}, lr={args.lr}, "
          f"seed={args.seed})")
    print(f"  rnn WAPE {_pct(rnn_w)} | gate {bt.GATE} {_pct(gate_w)} "
          f"| best baseline {best_n or 'n/a'} {_pct(best_w)}")
    if keep_decision(rnn_w, gate_w, best_w):
        margin = 100.0 * (best_w - rnn_w)
        print(f"  RNN VERDICT: KEEP - strictly beats both "
              f"(margin vs best baseline {margin:.2f} WAPE pts)")
        return 0
    reasons = []
    if rnn_w is None:
        reasons.append("no folds evaluated")
    else:
        if gate_w is not None and rnn_w >= gate_w:
            reasons.append(f"does not beat the naive gate ({_pct(gate_w)})")
        if best_w is not None and rnn_w >= best_w:
            reasons.append(
                f"does not beat the best baseline {best_n} ({_pct(best_w)})")
    print("  RNN VERDICT: DROP - must clearly beat BOTH gate and best "
          "baseline; ties lose (plan Week 6).")
    for r in reasons:
        print(f"    -> {r}")
    print("    -> valid result: revisit with real pilot data "
          "(plan sprint note 2026-11-13)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
