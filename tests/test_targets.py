#!/usr/bin/env python3
"""
test_targets.py — Unit tests for scripts/targets.py.

The expected labels are worked out by hand from the rows written in each
test; they are never taken from the function's own output.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.targets import LOOP_PASSES, build_target


def rows(pass_name, records):
    """Build a small label table: (inst_before, inst_control, inst_after, fired, outcome)."""
    return pd.DataFrame([
        {"pass_name": pass_name, "inst_before": before, "inst_control": control,
         "inst_after": after, "fired": fired, "outcome": outcome}
        for before, control, after, fired, outcome in records])


class TestOrdinaryPass(unittest.TestCase):
    """instcombine is not a loop pass: the reference is inst_before."""

    def setUp(self):
        self.df = rows("instcombine", [
            (100, 100, 90, 1, "decreased"),    # 10 % smaller
            (100, 100, 105, 1, "increased"),   # larger
            (100, 100, 100, 0, "equal"),       # untouched
            (200, 200, 199, 1, "decreased"),   # 0.5 % smaller: below the 1 % threshold
            (100, 100, None, None, "error"),   # opt failed
        ])

    def test_beneficial(self):
        clean, y, n_errors = build_target(self.df, "instcombine", "beneficial")
        self.assertEqual(n_errors, 1)
        self.assertEqual(len(clean), 4)
        self.assertEqual(y.tolist(), [1, 0, 0, 0])

    def test_beneficial_with_lower_threshold(self):
        # 1 / 200 = 0.5 % counts once the threshold is 0.5 %
        _, y, _ = build_target(self.df, "instcombine", "beneficial", threshold=0.005)
        self.assertEqual(y.tolist(), [1, 0, 0, 1])

    def test_harmful(self):
        _, y, _ = build_target(self.df, "instcombine", "harmful")
        self.assertEqual(y.tolist(), [0, 1, 0, 0])

    def test_fired(self):
        _, y, _ = build_target(self.df, "instcombine", "fired")
        self.assertEqual(y.tolist(), [1, 1, 0, 1])

    def test_input_is_not_modified(self):
        before = self.df.copy()
        build_target(self.df, "instcombine", "beneficial")
        pd.testing.assert_frame_equal(self.df, before)


class TestLoopPass(unittest.TestCase):
    """
    licm is a loop pass: the reference is inst_control, the size after loop
    canonicalization alone.  Here canonicalization adds 10 instructions.
    """

    def setUp(self):
        self.df = rows("licm", [
            (100, 110, 105, 1, "increased"),   # 5 below control: the pass itself helped
            (100, 110, 120, 1, "increased"),   # 10 above control: the pass itself hurt
            (100, 110, 110, 0, "increased"),   # equal to control: the pass did nothing
            (100, 100, 100, 0, "equal"),       # no loop, nothing happened
        ])

    def test_beneficial_is_measured_against_control(self):
        # (110 - 105) / 110 = 4.5 %  -> 1 ; the others are <= 0
        _, y, _ = build_target(self.df, "licm", "beneficial")
        self.assertEqual(y.tolist(), [1, 0, 0, 0])

    def test_harmful_is_measured_against_control(self):
        # Rows 1 and 3 are larger than inst_before, but only row 2 is larger
        # than the control.  Growth caused by canonicalization is not the pass's.
        _, y, _ = build_target(self.df, "licm", "harmful")
        self.assertEqual(y.tolist(), [0, 1, 0, 0])

    def test_loop_pass_list(self):
        self.assertEqual(LOOP_PASSES, {"licm", "loop-rotate", "indvars", "loop-deletion",
                                       "loop-idiom", "loop-unroll"})


class TestGuards(unittest.TestCase):
    def test_unknown_target(self):
        with self.assertRaises(ValueError):
            build_target(rows("gvn", [(10, 10, 9, 1, "decreased")]), "gvn", "useful")

    def test_rows_of_another_pass(self):
        with self.assertRaises(ValueError):
            build_target(rows("gvn", [(10, 10, 9, 1, "decreased")]), "licm", "fired")


if __name__ == "__main__":
    unittest.main(verbosity=2)
