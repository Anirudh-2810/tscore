"""Tests for reorder math + guards (plan Week 3 Done-when: every
suggestion respects the limits, and tests prove it)."""

import itertools
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import reorder as ro  # noqa: E402


class RawNeedTests(unittest.TestCase):
    def test_lead_time_and_safety_applied(self):
        # forecast 50/w, lead 2w, +20% -> 120 needed, 10 on hand -> 110
        g = ro.Guards(safety_buffer=0.2, lead_time_weeks=2)
        self.assertEqual(ro.raw_need(50, 10, g), 110)

    def test_surplus_stock_means_zero(self):
        g = ro.Guards()
        self.assertEqual(ro.raw_need(10, 100, g), 0)

    def test_rounding_ceils_partial_units(self):
        g = ro.Guards(safety_buffer=0.0, lead_time_weeks=1)
        self.assertEqual(ro.raw_need(10.2, 0, g), 11)   # ceil, never under-order

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            ro.raw_need(-1, 0, ro.Guards())
        with self.assertRaises(ValueError):
            ro.raw_need(5, -1, ro.Guards())
        with self.assertRaises(ValueError):
            ro.raw_need(5, 0, ro.Guards(lead_time_weeks=0))
        with self.assertRaises(ValueError):
            ro.raw_need(5, 0, ro.Guards(safety_buffer=-0.1))


class LineGuardTests(unittest.TestCase):
    def test_max_order_qty_caps(self):
        g = ro.Guards(max_order_qty=25)
        r = ro.suggested_qty(ro.Line("A", forecast=100, on_hand=0,
                                     unit_price=10.0), g)
        self.assertEqual(r.qty, 25)
        self.assertIn("max_order_qty", r.capped_by)

    def test_multiple_of_last_caps(self):
        g = ro.Guards(max_multiple_of_last=2.0)
        r = ro.suggested_qty(ro.Line("A", forecast=100, on_hand=0,
                                     unit_price=10.0, last_order_qty=10), g)
        self.assertEqual(r.qty, 20)                       # 10 * 2.0
        self.assertIn("max_multiple_of_last", r.capped_by)

    def test_multiple_guard_skipped_without_last_order(self):
        g = ro.Guards(max_multiple_of_last=2.0, safety_buffer=0.0)
        r = ro.suggested_qty(ro.Line("A", forecast=100, on_hand=0,
                                     unit_price=10.0, last_order_qty=None), g)
        self.assertEqual(r.qty, 100)                      # no ceiling to apply
        self.assertEqual(r.capped_by, [])

    def test_no_caps_returns_raw_need(self):
        g = ro.Guards(safety_buffer=0.25, lead_time_weeks=1)
        r = ro.suggested_qty(ro.Line("A", forecast=40, on_hand=5,
                                     unit_price=7.5), g)
        self.assertEqual(r.qty, 45)                       # ceil(50) - 5
        self.assertEqual(r.capped_by, [])
        self.assertAlmostEqual(r.spend, 45 * 7.5)


class BudgetTests(unittest.TestCase):
    def test_total_spend_never_exceeds_budget(self):
        g = ro.Guards(budget_cap=1000.0)
        lines = [ro.Line(f"I{i}", forecast=50, on_hand=0, unit_price=99.0)
                 for i in range(10)]
        out = ro.allocate(lines, g)
        self.assertLessEqual(sum(r.spend for r in out), 1000.0 + 1e-9)

    def test_urgent_line_filled_before_comfortable_one(self):
        # budget only covers one line -> urgent (2 weeks cover) wins whole,
        # comfortable (100 weeks cover) is truncated to whatever fits (0).
        g = ro.Guards(budget_cap=100.0, safety_buffer=0.0)
        urgent = ro.Line("URGENT", forecast=50, on_hand=0, unit_price=2.0)
        comfy = ro.Line("COMFY", forecast=1, on_hand=100, unit_price=2.0)
        out = {r.item: r for r in ro.allocate([comfy, urgent], g)}
        self.assertEqual(out["URGENT"].qty, 50)           # 50 * 2 = 100 fully fits
        self.assertLessEqual(out["COMFY"].qty, 0)         # nothing left
        self.assertLessEqual(sum(r.spend for r in out.values()), 100.0 + 1e-9)

    def test_partial_fill_of_binding_line(self):
        # budget 150, line costs 2/unit, raw need 100 -> can afford 75 only
        g = ro.Guards(budget_cap=150.0, safety_buffer=0.0)
        line = ro.Line("A", forecast=100, on_hand=0, unit_price=2.0)
        out = ro.allocate([line], g)
        self.assertEqual(out[0].qty, 75)
        self.assertIn("budget_cap", out[0].capped_by)
        self.assertLessEqual(out[0].spend, 150.0)

    def test_zero_budget_zeroes_everything(self):
        g = ro.Guards(budget_cap=0.0)
        out = ro.allocate([ro.Line("A", forecast=10, on_hand=0, unit_price=5.0)], g)
        self.assertEqual(out[0].qty, 0)

    def test_duplicate_items_rejected(self):
        g = ro.Guards()
        dup = [ro.Line("A", 1, 0, 1.0), ro.Line("A", 2, 0, 1.0)]
        with self.assertRaises(ValueError):
            ro.allocate(dup, g)


class LimitInvariantsTests(unittest.TestCase):
    """The Done-when: over a grid of inputs, NO suggestion ever breaks a
    configured limit."""

    def test_all_guards_hold_over_input_grid(self):
        forecasts = (0, 7, 33, 250)
        onHands = (0, 5, 80)
        lastOrders = (None, 0, 4, 30)
        budgets = (None, 0.0, 250.0, 10_000.0)
        for fc, oh, last, budget in itertools.product(
                forecasts, onHands, lastOrders, budgets):
            guards = ro.Guards(
                safety_buffer=0.2, lead_time_weeks=2,
                max_order_qty=60, max_multiple_of_last=2.5,
                budget_cap=budget)
            lines = [
                ro.Line(f"I-{fc}-{oh}-{last}", forecast=fc, on_hand=oh,
                        unit_price=13.0, last_order_qty=last),
                ro.Line(f"J-{fc}-{oh}-{last}", forecast=fc, on_hand=oh,
                        unit_price=7.0),
            ]
            out = ro.allocate(lines, guards)
            total = 0.0
            for r, line in zip(out, lines):
                self.assertGreaterEqual(r.qty, 0, (fc, oh, last, budget))
                self.assertLessEqual(r.qty, 60, "max_order_qty violated")
                if line.last_order_qty is not None:
                    self.assertLessEqual(
                        r.qty, int(2.5 * line.last_order_qty),
                        f"multiple guard violated: {(fc, oh, last, budget)}")
                total += r.spend
            if budget is not None:
                self.assertLessEqual(
                    total, budget + 1e-9,
                    f"budget violated: {(fc, oh, last, budget)}")


class DemoTests(unittest.TestCase):
    def test_demo_runs_and_respects_budget(self):
        self.assertEqual(ro.main(["--demo"]), 0)


if __name__ == "__main__":
    unittest.main()
