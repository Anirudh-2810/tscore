"""Tests for the weekly report (plan Week 4: a report an owner can read)."""

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import report as rp  # noqa: E402

MON = date(2026, 8, 3)  # a Monday


def weekly(item, first_week, quantities):
    return [(item, MON + timedelta(weeks=first_week + i), q)
            for i, q in enumerate(quantities)]


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


class ReportFixture(unittest.TestCase):
    """LIVE (hot seller, low stock) + DEAD (sold out weeks ago, stock stuck)
    + SHORT (3 weeks history -> naive fallback)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tscore_report_"))
        cls.sales = cls.tmp / "sales.csv"
        cls.stock = cls.tmp / "stock.csv"
        rows = []
        rows += weekly("LIVE-ITEM", 0, [20] * 10)
        rows += weekly("DEAD-ITEM", 0, [10] * 6)          # nothing last 4 weeks
        rows += weekly("SHORT-ITEM", 7, [4, 4, 4])        # only 3 weeks history
        write_csv(cls.sales, ["item", "date", "quantity", "price"], rows)
        write_csv(cls.stock, ["item", "on_hand", "unit_price", "last_order_qty"], [
            ["LIVE-ITEM", 2, 10.0, 20],
            ["DEAD-ITEM", 50, 5.0, 10],
            ["SHORT-ITEM", 0, 8.0, ""],
        ])

    def run_report(self, *extra):
        out_dir = self.tmp / ("out_" + str(abs(hash(extra))))
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = rp.main(["--sales", str(self.sales), "--stock", str(self.stock),
                            "--out-dir", str(out_dir), *extra])
        return code, out_dir, buf.getvalue()

    def test_report_and_forecast_written_and_readable(self):
        code, out, _ = self.run_report()
        self.assertEqual(code, 0)
        text = (out / "report.txt").read_text(encoding="utf-8")
        self.assertIn("ORDER THIS WEEK", text)
        self.assertIn("LIVE-ITEM", text)
        with open(out / "forecast.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, list(rp.FORECAST_HEADER))
            rows = {r["item"]: r for r in reader}
        self.assertIn("LIVE-ITEM", rows)
        # live item must be ordered (2 on hand, ~24 need) with valid confidence
        self.assertGreater(int(rows["LIVE-ITEM"]["suggested_reorder"]), 0)
        conf = float(rows["LIVE-ITEM"]["confidence"])
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        # dead item: forecast exists but stock is ample -> not ordered
        self.assertEqual(int(rows["DEAD-ITEM"]["suggested_reorder"]), 0)

    def test_alerts_and_fallback_notes(self):
        code, out, _ = self.run_report()
        self.assertEqual(code, 0)
        text = (out / "report.txt").read_text(encoding="utf-8")
        self.assertIn("RUNNING OUT", text)
        # live item is genuinely low: must be named in the running-out block
        block = text.split("RUNNING OUT")[1].split("DEAD STOCK")[0]
        self.assertIn("LIVE-ITEM", block)
        # dead item with 50 stuck units must be flagged
        dead_block = text.split("DEAD STOCK")[1]
        self.assertIn("DEAD-ITEM", dead_block)
        self.assertIn("DEAD-ITEM", dead_block.split("units stuck")[0])
        # short history falls back to naive and is disclosed
        self.assertIn("short history", text)
        self.assertIn("SHORT-ITEM", text.split("short history")[1])

    def test_budget_zero_disables_all_orders(self):
        code, out, _ = self.run_report("--budget", "1")
        self.assertEqual(code, 0)
        text = (out / "report.txt").read_text(encoding="utf-8")
        self.assertIn("ORDER THIS WEEK (0 items", text)
        # blocked demand must be disclosed, not silently dropped
        self.assertIn("WANTED BUT NOT ORDERED", text)
        blocked = text.split("WANTED BUT NOT ORDERED")[1].split("RUNNING OUT")[0]
        self.assertIn("LIVE-ITEM", blocked)
        self.assertIn("budget_cap", blocked)
        with open(out / "forecast.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.assertEqual(int(row["suggested_reorder"]), 0)

    def test_max_order_guard_flows_into_report(self):
        code, out, _ = self.run_report("--max-order", "3")
        self.assertEqual(code, 0)
        with open(out / "forecast.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.assertLessEqual(int(row["suggested_reorder"]), 3)

    def test_missing_stock_column_is_usage_error(self):
        bad = self.tmp / "bad_stock.csv"
        write_csv(bad, ["item", "on_hand"], [["X", 1]])
        with redirect_stdout(io.StringIO()):
            code = rp.main(["--sales", str(self.sales), "--stock", str(bad),
                            "--out-dir", str(self.tmp / "o1")])
        self.assertEqual(code, 1)

    def test_missing_sales_column_is_usage_error(self):
        bad = self.tmp / "bad_sales.csv"
        write_csv(bad, ["foo", "bar"], [["1", "2"]])
        with redirect_stdout(io.StringIO()):
            code = rp.main(["--sales", str(bad), "--stock", str(self.stock),
                            "--out-dir", str(self.tmp / "o2")])
        self.assertEqual(code, 1)


class StockLoadingTests(unittest.TestCase):
    def test_optional_last_order_and_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.csv"
            write_csv(p, ["item", "on_hand", "unit_price"], [["A", "5", "2.5"]])
            stock = rp.load_stock(str(p))
        self.assertEqual(stock["A"], (5, 2.5, None))

    def test_header_spaces_tolerated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.csv"
            write_csv(p, [" item ", " on_hand ", " unit_price "], [["A", "5", "2.5"]])
            stock = rp.load_stock(str(p))
        self.assertEqual(stock["A"], (5, 2.5, None))


class NoFoldHonestyTests(unittest.TestCase):
    """With too little history there is no measured error - the report must
    say the gate is not evaluable and must NOT claim confidence 1.0."""

    def test_gate_not_evaluable_and_placeholder_confidence(self):
        tmp = Path(tempfile.mkdtemp(prefix="tscore_nofold_"))
        sales = tmp / "sales.csv"
        stock = tmp / "stock.csv"
        write_csv(sales, ["item", "date", "quantity", "price"],
                  weekly("TINY", 0, [5, 5, 5]))
        write_csv(stock, ["item", "on_hand", "unit_price"], [["TINY", 1, 9.0]])
        out = tmp / "out"
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = rp.main(["--sales", str(sales), "--stock", str(stock),
                            "--out-dir", str(out)])
        self.assertEqual(code, 0)
        text = (out / "report.txt").read_text(encoding="utf-8")
        self.assertIn("GATE: not evaluable", text)
        self.assertIn("confidence = 0.5 placeholder", text)
        with open(out / "forecast.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(float(r["confidence"]), 0.5)


if __name__ == "__main__":
    unittest.main()
