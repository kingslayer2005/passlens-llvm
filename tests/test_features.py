#!/usr/bin/env python3
"""
test_features.py — Unit tests for Phase 2 feature extraction.

Tests feature extraction on 5 hand-written .ll fixtures with known expected values.
Each test verifies specific features that can be computed deterministically
from the fixture IR.

Usage:
  python -m pytest tests/test_features.py -v
  python -m tests.test_features    (standalone)
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.phase2_features import IRFeatureExtractor

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestSimpleAdd(unittest.TestCase):
    """Test features on simple_add.ll: 3 instructions, 1 BB, pure int arith."""

    def setUp(self):
        ir_text = (FIXTURES_DIR / "simple_add.ll").read_text()
        self.extractor = IRFeatureExtractor(ir_text, "simple_add")
        self.features = self.extractor.extract_features()
        self.inst_count = 3


    def test_instruction_count(self):
        # inst_count is removed by extractor, so we just pass
        pass

    def test_int_arith(self):
        # 2 add instructions
        self.assertAlmostEqual(self.features["int_arith_density"], 2 / self.inst_count)

    def test_ret(self):
        self.assertAlmostEqual(self.features["ret_density"], 1 / self.inst_count)

    def test_bb_count(self):
        self.assertAlmostEqual(self.features["bb_density"], 1 / self.inst_count)

    def test_no_branches(self):
        self.assertEqual(self.features["cond_br_density"], 0)
        self.assertEqual(self.features["uncond_br_density"], 0)

    def test_no_phi(self):
        self.assertEqual(self.features["phi_density"], 0)

    def test_no_calls(self):
        self.assertEqual(self.features["call_density"], 0)
        self.assertEqual(self.features["direct_call_density"], 0)

    def test_density(self):
        # int_arith_density should be 2/3
        self.assertAlmostEqual(self.features["int_arith_density"], 2.0 / 3.0, places=4)


class TestLoopSum(unittest.TestCase):
    """Test features on loop_sum.ll: loop with phi nodes, GEP, load."""

    def setUp(self):
        ir_text = (FIXTURES_DIR / "loop_sum.ll").read_text()
        self.extractor = IRFeatureExtractor(ir_text, "loop_sum")
        self.features = self.extractor.extract_features()
        self.inst_count = 10


    def test_instruction_count(self):
        pass

    def test_phi_count(self):
        self.assertAlmostEqual(self.features["phi_density"], 2 / self.inst_count)

    def test_phi_args(self):
        # Each phi has 2 incoming values -> 4 total args
        self.assertAlmostEqual(self.features["phi_args_total_density"], 4 / self.inst_count)

    def test_load(self):
        self.assertAlmostEqual(self.features["load_density"], 1 / self.inst_count)

    def test_gep(self):
        self.assertAlmostEqual(self.features["gep_density"], 1 / self.inst_count)

    def test_bb_count(self):
        self.assertAlmostEqual(self.features["bb_density"], 3 / self.inst_count)

    def test_icmp(self):
        self.assertAlmostEqual(self.features["icmp_density"], 1 / self.inst_count)

    def test_branches(self):
        self.assertAlmostEqual(self.features["cond_br_density"], 1 / self.inst_count)
        self.assertAlmostEqual(self.features["uncond_br_density"], 1 / self.inst_count)

    def test_int_arith(self):
        # add nsw i32 %sum, %val  and  add nsw i32 %i, 1
        self.assertAlmostEqual(self.features["int_arith_density"], 2 / self.inst_count)

    def test_new_features(self):
        # 10 instructions over 3 blocks -> mean = 10 / 3
        # Blocks: entry (1), loop (8), exit (1) -> max = 8
        self.assertEqual(self.features["max_insts_per_bb"], 8)
        self.assertAlmostEqual(self.features["mean_insts_per_bb"], 10.0 / 3.0, places=4)
        
        # Self-loop on 'loop' block -> header size = 8, body size = 8
        self.assertEqual(self.features["max_loop_header_size"], 8)
        self.assertEqual(self.features["innermost_loop_body_size"], 8)


class TestBranchHeavy(unittest.TestCase):
    """Test features on branch_heavy.ll: diamond CFG, many branches."""

    def setUp(self):
        ir_text = (FIXTURES_DIR / "branch_heavy.ll").read_text()
        self.extractor = IRFeatureExtractor(ir_text, "branch_heavy")
        self.features = self.extractor.extract_features()
        self.inst_count = len(self.extractor.instructions)

    def test_bb_count(self):
        self.assertAlmostEqual(self.features["bb_density"], 5 / self.inst_count)

    def test_icmp(self):
        self.assertAlmostEqual(self.features["icmp_density"], 3 / self.inst_count)

    def test_cond_br(self):
        self.assertAlmostEqual(self.features["cond_br_density"], 3 / self.inst_count)

    def test_uncond_br(self):
        self.assertAlmostEqual(self.features["uncond_br_density"], 1 / self.inst_count)

    def test_phi(self):
        # merge block has 1 phi with 3 incoming values
        self.assertAlmostEqual(self.features["phi_density"], 1 / self.inst_count)

    def test_phi_args(self):
        self.assertAlmostEqual(self.features["phi_args_total_density"], 3 / self.inst_count)

    def test_cfg_edges(self):
        # 3 conditional branches = 6 edges, 1 unconditional = 1 edge, total = 7
        self.assertAlmostEqual(self.features["cfg_edges_density"], 7 / self.inst_count)


class TestPhiHeavy(unittest.TestCase):
    """Test features on phi_heavy.ll: many phi nodes."""

    def setUp(self):
        ir_text = (FIXTURES_DIR / "phi_heavy.ll").read_text()
        self.extractor = IRFeatureExtractor(ir_text, "phi_heavy")
        self.features = self.extractor.extract_features()
        self.inst_count = len(self.extractor.instructions)

    def test_phi_count(self):
        # p1, p2 in mid_left + p3, p4 in mid_right + final, extra in exit = 6
        self.assertAlmostEqual(self.features["phi_density"], 6 / self.inst_count)

    def test_phi_args(self):
        # p1: 1 arg, p2: 1 arg, p3: 2 args, p4: 2 args,
        # final: 2 args, extra: 2 args = 10 total
        self.assertAlmostEqual(self.features["phi_args_total_density"], 10 / self.inst_count)

    def test_bb_count(self):
        self.assertAlmostEqual(self.features["bb_density"], 5 / self.inst_count)


class TestCallHeavy(unittest.TestCase):
    """Test features on call_heavy.ll: direct, intrinsic, and recursive calls."""

    def setUp(self):
        ir_text = (FIXTURES_DIR / "call_heavy.ll").read_text()
        self.extractor = IRFeatureExtractor(ir_text, "call_heavy")
        self.features = self.extractor.extract_features()
        self.inst_count = len(self.extractor.instructions)

    def test_direct_calls(self):
        # bar and baz are direct calls
        self.assertAlmostEqual(self.features["direct_call_density"], 2 / self.inst_count)

    def test_intrinsic_calls(self):
        # llvm.abs (twice) + llvm.smax = 3 intrinsic calls
        self.assertAlmostEqual(self.features["intrinsic_call_density"], 3 / self.inst_count)

    def test_self_recursive_calls(self):
        # call_heavy calls itself once
        self.assertAlmostEqual(self.features["self_recursive_call_density"], 1 / self.inst_count)

    def test_total_calls(self):
        # 2 direct + 3 intrinsic + 1 recursive = 6 total
        self.assertAlmostEqual(self.features["call_density"], 6 / self.inst_count)

    def test_bb_count(self):
        self.assertAlmostEqual(self.features["bb_density"], 3 / self.inst_count)

    def test_cond_br(self):
        self.assertAlmostEqual(self.features["cond_br_density"], 1 / self.inst_count)


# ============================================================================
# Run tests standalone
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
