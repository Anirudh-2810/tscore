"""Tests for the Week 8 demo driver (scripts/demo.py)."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import demo  # noqa: E402


class StepChainTests(unittest.TestCase):
    def test_default_chain_covers_the_four_shop_facing_stages(self):
        titles = [t for t, _, _ in demo.steps()]
        self.assertEqual(len(titles), 5)          # 2x gen + backtest + report + agents
        joined = " ".join(" ".join(a) for _, a, _ in demo.steps())
        self.assertIn("generate_fake_sales.py", joined)
        self.assertIn("backtest.py", joined)
        self.assertIn("report.py", joined)
        self.assertIn("agents_sim.py", joined)
        self.assertNotIn("rnn_test.py", joined)   # internal QA, not the demo

    def test_with_rnn_inserts_the_drop_tolerant_step(self):
        chain = demo.steps(with_rnn=True)
        rnn = [s for s in chain if "rnn_test.py" in " ".join(s[1])]
        self.assertEqual(len(rnn), 1)
        # DROP is exit 2 - the demo must accept it, not fail on it
        self.assertIn(2, rnn[0][2])

    def test_every_step_uses_scripts_under_the_repo(self):
        for _, argv, ok in demo.steps(with_rnn=True):
            self.assertTrue(argv[0].startswith("scripts/"), argv)
            self.assertTrue((ROOT / argv[0]).is_file(), argv)
            self.assertTrue(ok, argv)


class MainTests(unittest.TestCase):
    def test_dry_run_exits_zero_and_prints_artifacts(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = demo.main(["--dry-run"])
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("DEMO COMPLETE", text)
        self.assertIn("agents_dashboard.html", text)
        # dry-run must not claim it executed anything
        self.assertNotIn("FAILED", text)


if __name__ == "__main__":
    unittest.main()
