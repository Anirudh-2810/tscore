"""Contract tests for the fake-sales generator (data contract v1)."""

import csv
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import generate_fake_sales as gen  # noqa: E402

HEADER = ["item", "date", "quantity", "price"]


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tscore_"))
        cls.out = cls.tmp / "sales.csv"
        cls.weeks = 6
        cls.start = date(2026, 9, 1)
        rows = gen.generate(gen.DEFAULT_CATALOG[:8], cls.weeks, cls.start, seed=42)
        gen.write_csv(rows, cls.out)
        with open(cls.out, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cls.header = reader.fieldnames
            cls.rows = list(reader)

    def test_header_matches_contract(self):
        self.assertEqual(self.header, HEADER)

    def test_rows_parse_and_satisfy_contract(self):
        self.assertGreater(len(self.rows), 100, "expected a meaningful volume")
        last = (self.start + timedelta(days=self.weeks * 7)).isoformat()
        for r in self.rows:
            self.assertTrue(r["item"].strip())
            d = datetime.strptime(r["date"], "%Y-%m-%d").date()
            self.assertGreaterEqual(d, self.start)
            self.assertLess(d.isoformat(), last)
            self.assertGreaterEqual(int(r["quantity"]), 1)
            self.assertGreaterEqual(float(r["price"]), 0.0)

    def test_deterministic_for_same_seed(self):
        a = gen.generate(gen.DEFAULT_CATALOG[:4], 3, self.start, seed=7)
        b = gen.generate(gen.DEFAULT_CATALOG[:4], 3, self.start, seed=7)
        self.assertEqual(a, b)

    def test_seed_changes_output(self):
        a = gen.generate(gen.DEFAULT_CATALOG[:4], 3, self.start, seed=1)
        b = gen.generate(gen.DEFAULT_CATALOG[:4], 3, self.start, seed=2)
        self.assertNotEqual(a, b)

    def test_patterns_present(self):
        """Weekend and seasonal signal must actually show up in the data."""
        weekend = weekday = 0
        monthly = {}
        for r in self.rows:
            d = datetime.strptime(r["date"], "%Y-%m-%d").date()
            qty = int(r["quantity"])
            if d.weekday() >= 5:
                weekend += qty
            else:
                weekday += qty
            monthly[d.strftime("%Y-%m")] = monthly.get(d.strftime("%Y-%m"), 0) + qty
        # weekend uplift: per-day weekend sales exceed per-day weekday sales
        self.assertGreater(weekend / (self.weeks * 7 * 2 / 7),
                           weekday / (self.weeks * 7 * 5 / 7) * 0.9)
        # seasonal: month totals are not flat
        self.assertGreater(max(monthly.values()), min(monthly.values()) * 1.05)

    def test_real_format_sample_is_contract_valid(self):
        sample = ROOT / "examples" / "real_format_sales.csv"
        with open(sample, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, HEADER)
            for r in reader:
                datetime.strptime(r["date"], "%Y-%m-%d")
                self.assertGreaterEqual(int(r["quantity"]), 1)
                self.assertGreaterEqual(float(r["price"]), 0.0)


if __name__ == "__main__":
    unittest.main()
