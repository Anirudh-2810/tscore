"""Tests for the walk-forward backtest (Week 2)."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import backtest as bt  # noqa: E402


def week_rows(item, start, quantities):
    """One sales row per week on the given Mondays."""
    return [(item, start + timedelta(weeks=i), q)
            for i, q in enumerate(quantities)]


class WeeklySeriesTests(unittest.TestCase):
    def test_bucketing_zero_fill_and_pre_first_none(self):
        mon = date(2026, 9, 7)  # a Monday
        rows = [
            ("A", mon + timedelta(days=0), 2),        # week 0
            ("A", mon + timedelta(days=6), 3),        # week 0 (Sunday)
            ("A", mon + timedelta(days=9), 1),        # week 1
            # week 2: no sales -> 0.0 after first sale
            ("B", mon + timedelta(days=14), 7),       # B first appears week 2
        ]
        weeks, series = bt.weekly_series(rows)
        self.assertEqual(weeks, [mon, mon + timedelta(weeks=1), mon + timedelta(weeks=2)])
        self.assertEqual(series["A"], [5.0, 1.0, 0.0])
        # B must be None (not 0) in weeks before its introduction
        self.assertEqual(series["B"], [None, None, 7.0])

    def test_axis_spans_global_range(self):
        mon = date(2026, 9, 7)
        rows = [("A", mon, 1), ("A", mon + timedelta(weeks=5), 1)]
        weeks, series = bt.weekly_series(rows)
        self.assertEqual(len(weeks), 6)
        self.assertEqual(series["A"][3], 0.0)  # gap weeks are zero


class ForecasterTests(unittest.TestCase):
    def test_naive_seasonal_ma(self):
        hist = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        self.assertEqual(bt.f_naive_last_week(hist), 8.0)
        self.assertEqual(bt.f_seasonal_naive(4)(hist), 4.0)   # same week last month
        self.assertEqual(bt.f_moving_average(4)(hist), 6.5)   # mean(5,6,7,8)
        self.assertEqual(bt.f_moving_average(2)(hist), 7.5)

    def test_required_history_tracks_models(self):
        models = bt.build_models([2, 8], 4)
        self.assertEqual(bt.required_history(models), 8)      # max(k, lag+1)
        self.assertEqual(bt.required_history(bt.build_models([2], 4)), 5)


class StockSimTests(unittest.TestCase):
    def test_surplus_carries_and_no_stockout(self):
        # init 8; order 10 covers demand 8 (leftover 10); order 4+10 >= 12
        self.assertEqual(bt.run_stock([(10.0, 8.0), (4.0, 12.0)], 8), 0)

    def test_stockout_when_order_and_stock_cannot_cover(self):
        # init 2 + order 3 = 5 < demand 10 -> stockout, stock resets to 0
        self.assertEqual(bt.run_stock([(3.0, 10.0)], 2), 1)

    def test_forecast_is_rounded_for_orders(self):
        # pred 3.4 rounds to 3; 2+3=5 < 6 -> stockout
        self.assertEqual(bt.run_stock([(3.4, 6.0)], 2), 1)
        # pred 3.6 rounds to 4; 2+4=6 >= 6 -> fine
        self.assertEqual(bt.run_stock([(3.6, 6.0)], 2), 0)


class LeakageTests(unittest.TestCase):
    def test_future_data_cannot_change_past_predictions(self):
        mon = date(2026, 9, 7)
        base = [5] * 8
        short_rows = week_rows("X", mon, base)                 # weeks 0..7
        full_rows = week_rows("X", mon, base + [100, 100])     # weeks 0..9
        weeks_s, series_s = bt.weekly_series(short_rows)
        weeks_f, series_f = bt.weekly_series(full_rows)
        models = [(bt.GATE, bt.f_naive_last_week)]
        stats_s, preds_s = bt.evaluate(weeks_s, series_s, models, min_hist=2)
        stats_f, preds_f = bt.evaluate(weeks_f, series_f, models, min_hist=2)
        past_s = {w: p for _, w, p, _ in preds_s}
        past_f = {w: p for _, w, p, _ in preds_f}
        self.assertTrue(past_s)                                # folds existed
        for key, pred in past_s.items():
            self.assertEqual(past_f[key], pred,
                             f"prediction for week {key} changed")
        # and the full run must actually have more folds (future got used later)
        self.assertGreater(len(preds_f), len(preds_s))


class VerdictTests(unittest.TestCase):
    @staticmethod
    def _stats(gate_wape, model_wape):
        def mk(w):
            return {"ae": w, "act": 1.0, "se": 0.0, "n": 1.0,
                    "under": 0.0, "stockouts": 0.0}
        return {bt.GATE: mk(gate_wape),
                "moving_average(k=4)": mk(model_wape)}

    def test_gate_pass_when_model_beats_naive(self):
        winner, _, passed = bt.verdict(self._stats(0.20, 0.10))
        self.assertTrue(passed)
        self.assertEqual(winner, "moving_average(k=4)")

    def test_gate_fail_when_no_model_beats_naive(self):
        winner, _, passed = bt.verdict(self._stats(0.10, 0.20))
        self.assertFalse(passed)


class EndToEndTests(unittest.TestCase):
    def test_backtest_runs_on_committed_fake_data(self):
        """Gate result is data-dependent (0 or 2); crashes are bugs (1)."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = bt.main(["--sales", str(ROOT / "examples" / "fake_sales.csv")])
        self.assertIn(code, (0, 2), f"unexpected exit {code}: {buf.getvalue()[-500:]}")
        self.assertIn("GATE VERDICT", buf.getvalue())

    def test_bad_input_is_usage_error(self):
        with redirect_stdout(io.StringIO()):
            code = bt.main(["--sales", str(ROOT / "README.md")])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
