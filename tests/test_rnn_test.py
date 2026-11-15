"""Tests for the Week 6 RNN experiment (scripts/rnn_test.py)."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import backtest as bt  # noqa: E402
import rnn_test as rt  # noqa: E402


def synth_rows(n_weeks=16):
    """Two deterministic items, one row per week, starting the same Monday."""
    mon = date(2026, 9, 7)
    rows = []
    for item, base in (("A", 5), ("B", 9)):
        for w in range(n_weeks):
            rows.append((item, mon + timedelta(weeks=w), base + (w % 3)))
    return rows


class FoldParityTests(unittest.TestCase):
    def test_folds_match_backtest_evaluate(self):
        weeks, series = bt.weekly_series(synth_rows())
        min_hist = bt.required_history(bt.build_models([2, 4, 8], 4))
        mine = {(i, t + 1) for i, t, _, _ in rt.iter_folds(series, weeks, min_hist)}
        stats, preds = bt.evaluate(
            weeks, series, [(bt.GATE, bt.f_naive_last_week)], min_hist)
        theirs = {(i, t) for i, t, _, _ in preds}
        self.assertEqual(mine, theirs)
        self.assertGreater(stats[bt.GATE]["n"], 0)

    def test_iter_folds_hist_is_prefix_of_series(self):
        weeks, series = bt.weekly_series(synth_rows())
        for item, cut, hist, actual in rt.iter_folds(series, weeks, 8):
            vals = series[item]
            self.assertEqual(hist, [v for v in vals[:cut + 1] if v is not None])
            self.assertEqual(actual, vals[cut + 1])


class NoLeakageTests(unittest.TestCase):
    def test_train_samples_exclude_future_and_scale_too(self):
        # weeks 0..4 are real, week 5 is the eval target (sentinel 999),
        # week 6 is further future (777) - neither may appear in training.
        series = {"A": [4.0, 5.0, 6.0, 7.0, 8.0, 999.0, 777.0]}
        samples = rt.train_samples(series, cut=4, min_hist=2)
        self.assertTrue(samples)
        flat_targets = [t for _, t in samples]
        flat_hist = [v for h, _ in samples for v in h]
        self.assertNotIn(999.0, flat_targets)
        self.assertNotIn(999.0, flat_hist)
        self.assertNotIn(777.0, flat_targets)
        self.assertNotIn(777.0, flat_hist)
        self.assertLessEqual(max(flat_targets), 8.0)
        # scale is train-fold stats only: future sentinel must not inflate it
        memo = {}
        self.assertEqual(rt.scale_at(series, 4, memo), 8.0)
        self.assertEqual(rt.scale_at(series, 6, memo), 999.0)


class ModelTests(unittest.TestCase):
    def test_determinism_same_seed_same_predictions(self):
        samples = [rt.to_seq([4.0, 5.0, 6.0, 7.0, 8.0], 9.0, 8.0)]
        a = rt.ElmanRNN(hidden=6, seed=7, lr=0.02)
        b = rt.ElmanRNN(hidden=6, seed=7, lr=0.02)
        a.fit(samples, epochs=5, seed=42)
        b.fit(samples, epochs=5, seed=42)
        xs = [4.0 / 8.0, 5.0 / 8.0, 6.0 / 8.0]
        self.assertEqual(a.predict(xs), b.predict(xs))

    def test_clone_is_independent(self):
        m = rt.ElmanRNN(hidden=4, seed=1, lr=0.02)
        c = m.clone()
        m.wx[0] = 999.0
        self.assertNotEqual(c.wx[0], 999.0)

    def test_rnn_predict_clamps_at_zero(self):
        snap = rt.ElmanRNN(hidden=4, seed=1, lr=0.02)
        snap.bo = -1000.0                    # force a negative raw output
        self.assertEqual(rt.rnn_predict(snap, [5.0, 6.0], scale=6.0), 0.0)

    def test_snapshots_cover_cuts_and_are_distinct(self):
        weeks, series = bt.weekly_series(synth_rows())
        # sentinel: make week 10+ huge; snapshots for cut < 10 must be built
        # from train_samples (tested above) - here we check the snapshots
        # dict covers exactly the requested cuts and models are distinct.
        cuts = [8, 9, 10]
        snaps = rt.train_snapshots(series, cuts, min_hist=8, epochs=1,
                                   hidden=4, lr=0.02, seed=3)
        self.assertEqual(sorted(snaps), cuts)
        self.assertFalse(snaps[8] is snaps[10])


class VerdictTests(unittest.TestCase):
    def test_keep_needs_strict_win_on_both(self):
        self.assertTrue(rt.keep_decision(25.0, 34.0, 28.0))   # beats both
        self.assertFalse(rt.keep_decision(28.0, 34.0, 28.0))  # tie = lose
        self.assertFalse(rt.keep_decision(30.0, 28.0, 26.0))  # worse
        self.assertFalse(rt.keep_decision(25.0, 34.0, None))  # no baseline
        self.assertFalse(rt.keep_decision(None, 34.0, 28.0))  # no folds


class IntegrationTests(unittest.TestCase):
    def test_main_runs_and_prints_verdict(self):
        sales = ROOT / "examples" / "fake_sales.csv"
        if not sales.exists():
            self.skipTest("examples/fake_sales.csv missing")
        out = io.StringIO()
        with redirect_stdout(out):
            rc = rt.main(["--sales", str(sales), "--epochs", "2",
                          "--seed", "7"])
        text = out.getvalue()
        self.assertIn(rc, (0, 2), f"unexpected exit {rc}\n{text}")
        self.assertIn("Walk-forward backtest", text)
        self.assertIn("RNN VERDICT", text)
        self.assertIn("elman_rnn", text)


if __name__ == "__main__":
    unittest.main()
