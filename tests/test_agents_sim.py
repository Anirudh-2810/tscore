"""Tests for the Week 7 live-and-die agent simulation (scripts/agents_sim.py)."""

import io
import json
import random
import sys
import unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import agents_sim as ag  # noqa: E402
import backtest as bt  # noqa: E402


def week_rows(item, start, quantities):
    return [(item, start + timedelta(weeks=i), q)
            for i, q in enumerate(quantities)]


class ForecastTests(unittest.TestCase):
    def test_uses_past_only_and_empty_hist_is_zero(self):
        # forecast() takes hist explicitly; the sim builds hist = vals[:t]
        # (tested via run_sim leakage below).
        self.assertEqual(ag.forecast([], 8), 0.0)
        self.assertEqual(ag.forecast([2.0, 4.0, 6.0], 2), 5.0)
        self.assertEqual(ag.forecast([2.0, 4.0, 6.0], 8), 4.0)


class StepAgentTests(unittest.TestCase):
    def test_hand_computed_week(self):
        # hist all 5, k=8, safety 0 -> target 5; order 5 (cost 5);
        # demand 7 -> sold 5 (revenue 5*1.2=6), lost 2 (penalty 4);
        # no leftover; upkeep 10. wallet = 100 + 6 - 5 - 4 - 10 = 87.
        a = ag.Agent(id="T", lineage="L", gen=0, parent=None, ma_k=8,
                     safety=0.0, wallet=100.0, born_week=0)
        ag.step_agent(a, {"X": 7.0}, {"X": [5.0] * 8})
        self.assertAlmostEqual(a.wallet, 87.0)
        self.assertAlmostEqual(a.lost_units, 2.0)
        self.assertEqual(a.stockout_item_weeks, 1)
        self.assertAlmostEqual(a.stock["X"], 0.0)

    def test_overstock_holding_drains_wallet(self):
        a = ag.Agent(id="T", lineage="L", gen=0, parent=None, ma_k=8,
                     safety=0.0, wallet=100.0, born_week=0)
        ag.step_agent(a, {"X": 0.0}, {"X": [10.0] * 8})
        # order 10 (cost 10), no sales, leftover 10 * 0.35 = 3.5 holding,
        # upkeep 10: 100 - 10 - 3.5 - 10 = 76.5
        self.assertAlmostEqual(a.wallet, 76.5)
        self.assertAlmostEqual(a.stock["X"], 10.0)

    def test_order_capped_at_max_order(self):
        a = ag.Agent(id="T", lineage="L", gen=0, parent=None, ma_k=8,
                     safety=0.6, wallet=1000.0, born_week=0)
        ag.step_agent(a, {"X": 0.0}, {"X": [100.0] * 8})
        self.assertLessEqual(a.stock["X"], ag.MAX_ORDER)


class MutationTests(unittest.TestCase):
    def test_always_changes_at_least_one_gene(self):
        rng = random.Random(3)
        for k in (1, 5, 12):
            for s in (0.0, 0.3, 0.6):
                for _ in range(30):
                    self.assertNotEqual(ag.mutate(k, s, rng), (k, s))

    def test_stays_in_bounds(self):
        rng = random.Random(9)
        for _ in range(200):
            k, s = ag.mutate(1, 0.0, rng)
            self.assertTrue(ag.K_RANGE[0] <= k <= ag.K_RANGE[1])
            self.assertTrue(ag.SAFETY_RANGE[0] <= s <= ag.SAFETY_RANGE[1])


class SimLifecycleTests(unittest.TestCase):
    def test_death_at_zero_and_immigrant_refill(self):
        # demand 60 with no history: agents can't order, take the stockout
        # hit every week -> everyone dies at week 0, house money refills.
        mon = date(2026, 9, 7)
        rows = week_rows("X", mon, [60] * 10)
        weeks, series = bt.weekly_series(rows)
        res = ag.run_sim(weeks, series, seed=1, initial_wealth=5.0,
                         spawn_cost=40.0, pop_min=3, max_pop=6)
        kinds = [e["kind"] for e in res["events"] if e["week"] == 0]
        self.assertIn("DEAD", kinds)
        self.assertIn("IMMIGRANT", kinds)
        self.assertIn("LINEAGE_EXTINCT", kinds)
        week0 = next(e for e in res["events"]
                     if e["kind"] == "WEEK" and e["week"] == 0)
        self.assertEqual(week0["pop"], 3)   # refilled to pop_min

    def test_spawn_money_comes_from_parent(self):
        mon = date(2026, 9, 7)
        # flat demand 5: agents order steadily, stay solvent, rich enough
        # to spawn at spawn_cost.
        rows = week_rows("X", mon, [5] * 16)
        weeks, series = bt.weekly_series(rows)
        res = ag.run_sim(weeks, series, seed=2, initial_wealth=1000.0,
                         spawn_cost=40.0, pop_min=4, max_pop=12)
        spawns = [e for e in res["events"] if e["kind"] == "SPAWN"]
        self.assertTrue(spawns, "expected at least one spawn")
        by_id = {a["id"]: a for a in res["roster"]}
        for e in spawns[:3]:
            child = by_id[e["id"]]
            parent = by_id[e["parent"]]
            # spawn-time gift (event), not end-of-run wallet: roster stores
            # final state after the child kept trading
            self.assertEqual(e["wallet"], 40.0)
            self.assertEqual(child["gen"], parent["gen"] + 1)
            self.assertEqual(child["lineage"], parent["lineage"])
            self.assertNotEqual((child["ma_k"], child["safety"]),
                                (parent["ma_k"], parent["safety"]))
            # spawn_cost is recorded as the gift the parent paid out
            self.assertEqual(e["wallet"], ag.SPAWN_COST)

    def test_no_future_leakage_between_twin_series(self):
        mon = date(2026, 9, 7)
        base = [4, 5, 6, 4, 5, 6, 4, 5, 6, 4]
        rows_a = week_rows("X", mon, base)
        rows_b = week_rows("X", mon, base[:5] + [999] + base[6:])
        weeks_a, series_a = bt.weekly_series(rows_a)
        weeks_b, series_b = bt.weekly_series(rows_b)
        res_a = ag.run_sim(weeks_a, series_a, seed=5, pop_min=4, max_pop=8)
        res_b = ag.run_sim(weeks_b, series_b, seed=5, pop_min=4, max_pop=8)

        def early(res):
            return [e for e in res["events"]
                    if e["kind"] in ("SPAWN", "DEAD", "IMMIGRANT")
                    and e["week"] < 5]
        self.assertEqual(early(res_a), early(res_b))

    def test_determinism_same_seed(self):
        mon = date(2026, 9, 7)
        rows = week_rows("X", mon, [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6])
        weeks, series = bt.weekly_series(rows)
        a = ag.run_sim(weeks, series, seed=7, pop_min=4, max_pop=10)
        b = ag.run_sim(weeks, series, seed=7, pop_min=4, max_pop=10)
        self.assertEqual(json.dumps(a["events"]), json.dumps(b["events"]))


class SummaryTests(unittest.TestCase):
    def test_lineage_improvement_metrics_present(self):
        mon = date(2026, 9, 7)
        rows = week_rows("X", mon, [5] * 16)
        weeks, series = bt.weekly_series(rows)
        res = ag.run_sim(weeks, series, seed=2, initial_wealth=1000.0,
                         pop_min=4, max_pop=12)
        s = res["summary"]
        self.assertIn("by_gen", s)
        self.assertIn("descendants_better", s)
        self.assertGreaterEqual(s["lineages_total"], 1)
        self.assertGreater(s["agents_born"], 4)


class OutputTests(unittest.TestCase):
    def test_log_and_html_render(self):
        mon = date(2026, 9, 7)
        rows = week_rows("X", mon, [5] * 12)
        weeks, series = bt.weekly_series(rows)
        res = ag.run_sim(weeks, series, seed=7, pop_min=4, max_pop=10)
        log = ag.render_log(res)
        self.assertIn("by generation", log)
        self.assertIn("lineage check", log)
        html = ag.render_html(res)
        self.assertIn("live-and-die agents", html)
        self.assertIn("event stream", html)
        self.assertIn("const DATA =", html)
        # JSON payload must be parseable (dashboard data contract)
        payload = html.split("const DATA = ", 1)[1].split(";", 1)[0]
        json.loads(payload)

    def test_main_writes_files(self):
        mon = date(2026, 9, 7)
        rows = week_rows("X", mon, [5] * 12)
        weeks, series = bt.weekly_series(rows)
        sales = ROOT / "examples" / "fake_sales.csv"
        if not sales.exists():
            self.skipTest("examples/fake_sales.csv missing")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out_html = Path(td) / "dash.html"
            out_log = Path(td) / "run.log"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = ag.main(["--sales", str(sales), "--html", str(out_html),
                              "--log", str(out_log), "--pop", "4"])
            self.assertEqual(rc, 0, buf.getvalue())
            self.assertTrue(out_html.exists())
            self.assertIn("live-and-die agents",
                          out_html.read_text(encoding="utf-8"))
            self.assertTrue(out_log.exists())
            log_text = out_log.read_text(encoding="utf-8")
            self.assertIn("by generation", log_text)
            self.assertIn("lineage check", log_text)


if __name__ == "__main__":
    unittest.main()
