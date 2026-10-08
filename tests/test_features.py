#!/usr/bin/env python3
"""
test_features.py — Unit tests for the IR parser and the feature extractor.

Every expected value below was worked out BY HAND from the fixture file
(the instruction-by-instruction count is written next to each test class).
The expected values are never taken from the extractor's own output.

Usage:
  python -m unittest tests.test_features -v
  python -m pytest tests/test_features.py -v
"""

import shutil
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.phase2_features import (
    IRFeatureExtractor, parse_loop_info, reachable_counts,
)
from scripts.utils import count_instructions, hash_ir, normalize_ir

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def load(name):
    """Read a fixture and return (ir_text, extractor)."""
    ir_text = (FIXTURES_DIR / f"{name}.ll").read_text()
    return ir_text, IRFeatureExtractor(ir_text, name)


class TestSimpleAdd(unittest.TestCase):
    """
    simple_add.ll
      entry: add, add, ret                       -> 3 instructions, 1 block
    """

    def setUp(self):
        self.ir_text, self.extractor = load("simple_add")
        self.features = self.extractor.extract_features(loops=[])

    def test_instruction_count(self):
        self.assertEqual(count_instructions(self.ir_text), 3)
        self.assertEqual(len(self.extractor.instructions), 3)

    def test_densities(self):
        self.assertAlmostEqual(self.features["int_arith_density"], 2 / 3)
        self.assertAlmostEqual(self.features["ret_density"], 1 / 3)
        self.assertAlmostEqual(self.features["bb_density"], 1 / 3)
        self.assertEqual(self.features["cond_br_density"], 0)
        self.assertEqual(self.features["uncond_br_density"], 0)
        self.assertEqual(self.features["phi_density"], 0)
        self.assertEqual(self.features["call_density"], 0)

    def test_constants(self):
        # only "add nsw i32 %sum, 1" has a constant operand
        self.assertAlmostEqual(self.features["const_operands_density"], 1 / 3)

    def test_no_loops(self):
        self.assertEqual(self.features["loop_count_density"], 0)
        self.assertEqual(self.features["max_loop_depth"], 0)
        self.assertEqual(self.features["innermost_loop_body_size"], 0)

    def test_no_raw_counts(self):
        # raw counts must not be features (they are proxies for size)
        for name in self.features:
            self.assertFalse(name.endswith("_count") and name != "log_inst_count", name)


class TestLoopSum(unittest.TestCase):
    """
    loop_sum.ll
      entry: br                                   -> 1
      loop:  phi, phi, getelementptr, load,
             add, add, icmp, br                   -> 8
      exit:  ret                                  -> 1
      total 10 instructions, 3 blocks, 1 loop (the block "loop" on its own)
    """

    def setUp(self):
        self.ir_text, self.extractor = load("loop_sum")
        # What LoopInfo prints for this function
        self.loops = parse_loop_info("Loop at depth 1 containing: %loop<header><latch><exiting>")
        self.features = self.extractor.extract_features(loops=self.loops)

    def test_instruction_count(self):
        self.assertEqual(count_instructions(self.ir_text), 10)

    def test_densities(self):
        self.assertAlmostEqual(self.features["phi_density"], 2 / 10)
        self.assertAlmostEqual(self.features["phi_args_total_density"], 4 / 10)
        self.assertAlmostEqual(self.features["load_density"], 1 / 10)
        self.assertAlmostEqual(self.features["gep_density"], 1 / 10)
        self.assertAlmostEqual(self.features["bb_density"], 3 / 10)
        self.assertAlmostEqual(self.features["icmp_density"], 1 / 10)
        self.assertAlmostEqual(self.features["cond_br_density"], 1 / 10)
        self.assertAlmostEqual(self.features["uncond_br_density"], 1 / 10)
        self.assertAlmostEqual(self.features["int_arith_density"], 2 / 10)

    def test_constants(self):
        # two phi nodes with incoming 0, and "add nsw i32 %i, 1"
        self.assertAlmostEqual(self.features["const_operands_density"], 3 / 10)

    def test_block_sizes(self):
        self.assertEqual(self.features["max_insts_per_bb"], 8)
        self.assertAlmostEqual(self.features["mean_insts_per_bb"], 10 / 3)

    def test_loop_features(self):
        self.assertAlmostEqual(self.features["loop_count_density"], 1 / 10)
        self.assertEqual(self.features["max_loop_depth"], 1)
        self.assertEqual(self.features["max_loop_header_size"], 8)
        self.assertEqual(self.features["min_loop_header_size"], 8)
        self.assertEqual(self.features["innermost_loop_body_size"], 8)


class TestBranchHeavy(unittest.TestCase):
    """
    branch_heavy.ll
      entry: icmp, br            -> 2     then1: add, icmp, br   -> 3
      else1: sub, icmp, br       -> 3     then2: mul, br         -> 2
      merge: phi, ret            -> 2
      total 12 instructions, 5 blocks
      edges: entry->then1, entry->else1, then1->then2, then1->merge,
             else1->then2, else1->merge, then2->merge            -> 7
    """

    def setUp(self):
        self.ir_text, self.extractor = load("branch_heavy")
        self.features = self.extractor.extract_features(loops=[])

    def test_instruction_count(self):
        self.assertEqual(count_instructions(self.ir_text), 12)

    def test_densities(self):
        self.assertAlmostEqual(self.features["bb_density"], 5 / 12)
        self.assertAlmostEqual(self.features["icmp_density"], 3 / 12)
        self.assertAlmostEqual(self.features["cond_br_density"], 3 / 12)
        self.assertAlmostEqual(self.features["uncond_br_density"], 1 / 12)
        self.assertAlmostEqual(self.features["phi_density"], 1 / 12)
        self.assertAlmostEqual(self.features["phi_args_total_density"], 3 / 12)
        self.assertAlmostEqual(self.features["int_arith_density"], 3 / 12)   # add, sub, mul
        self.assertAlmostEqual(self.features["cfg_edges_density"], 7 / 12)

    def test_predecessors_and_successors(self):
        # predecessors: entry 0, then1 1, else1 1, then2 2, merge 3
        self.assertAlmostEqual(self.features["blocks_1_pred_density"], 2 / 12)
        self.assertAlmostEqual(self.features["blocks_2_pred_density"], 1 / 12)
        self.assertAlmostEqual(self.features["blocks_gt2_pred_density"], 1 / 12)
        # successors: entry 2, then1 2, else1 2, then2 1, merge 0
        self.assertAlmostEqual(self.features["blocks_1_succ_density"], 1 / 12)
        self.assertAlmostEqual(self.features["blocks_2_succ_density"], 3 / 12)

    def test_constants(self):
        # 10, 1, 20, 1, 0, 2
        self.assertAlmostEqual(self.features["const_operands_density"], 6 / 12)


class TestPhiHeavy(unittest.TestCase):
    """
    phi_heavy.ll
      entry:     icmp, br          -> 2     left:      add, icmp, br  -> 3
      right:     sub, br           -> 2     mid_left:  phi, phi, br   -> 3
      mid_right: phi, phi, br      -> 3     exit: phi, phi, add, ret  -> 4
      total 17 instructions, 6 blocks, 6 phi nodes
      phi incoming values: 1 + 1 + 2 + 2 + 2 + 2 = 10
    """

    def setUp(self):
        self.ir_text, self.extractor = load("phi_heavy")
        self.features = self.extractor.extract_features(loops=[])

    def test_instruction_count(self):
        self.assertEqual(count_instructions(self.ir_text), 17)

    def test_densities(self):
        self.assertAlmostEqual(self.features["phi_density"], 6 / 17)
        self.assertAlmostEqual(self.features["phi_args_total_density"], 10 / 17)
        self.assertAlmostEqual(self.features["bb_density"], 6 / 17)
        self.assertAlmostEqual(self.features["const_operands_density"], 4 / 17)  # 0, 1, 5, 1


class TestCallHeavy(unittest.TestCase):
    """
    call_heavy.ll
      entry: icmp, br                               -> 2
      then:  call bar, call baz, call llvm.abs,
             call llvm.smax, call call_heavy, ret   -> 6
      else:  sub, call llvm.abs, ret                -> 3
      total 11 instructions, 3 blocks
      calls: 2 direct, 3 intrinsic, 1 self-recursive = 6
    """

    def setUp(self):
        self.ir_text, self.extractor = load("call_heavy")
        self.features = self.extractor.extract_features(loops=[])

    def test_instruction_count(self):
        self.assertEqual(count_instructions(self.ir_text), 11)

    def test_densities(self):
        self.assertAlmostEqual(self.features["direct_call_density"], 2 / 11)
        self.assertAlmostEqual(self.features["intrinsic_call_density"], 3 / 11)
        self.assertAlmostEqual(self.features["self_recursive_call_density"], 1 / 11)
        self.assertAlmostEqual(self.features["call_density"], 6 / 11)
        self.assertAlmostEqual(self.features["bb_density"], 3 / 11)
        self.assertAlmostEqual(self.features["cond_br_density"], 1 / 11)
        self.assertAlmostEqual(self.features["ret_density"], 2 / 11)

    def test_constants(self):
        # icmp ... 0 | i1 true | i32 0 (smax) | sub ... 0 | i1 true
        self.assertAlmostEqual(self.features["const_operands_density"], 5 / 11)


# A function written the way clang prints it: numeric labels, a multi-line
# switch, bitwise and shift instructions, and one unreachable block.
#   entry (no label): and, shl, switch              -> 3
#   4:  br                                          -> 1
#   5:  xor, br                                     -> 2
#   7:  br            (no predecessors: dead code)  -> 1
#   8:  phi, ret                                    -> 2
#   total 9 instructions in 5 blocks; 8 instructions in 4 reachable blocks
CLANG_STYLE_IR = """
define i32 @clang_style(i32 %0, i32 %1) #0 {
  %3 = and i32 %0, 255
  %x = shl i32 %3, 2
  switch i32 %x, label %8 [
    i32 0, label %4
    i32 4, label %5
  ]

4:                                                ; preds = %2
  br label %8

5:                                                ; preds = %2
  %6 = xor i32 %1, -1
  br label %8

7:                                                ; No predecessors!
  br label %8

8:                                                ; preds = %7, %5, %4, %2
  %9 = phi i32 [ 1, %4 ], [ %6, %5 ], [ 0, %7 ], [ %x, %2 ]
  ret i32 %9
}
"""


class TestClangStyleIR(unittest.TestCase):
    """Numeric labels, switch tables and dead blocks must be handled."""

    def setUp(self):
        self.extractor = IRFeatureExtractor(CLANG_STYLE_IR, "clang_style")
        self.features = self.extractor.extract_features(loops=[])

    def test_labels_are_not_instructions(self):
        # 9 instructions; the 4 numeric labels and the 3 switch-table lines
        # are not instructions
        self.assertEqual(count_instructions(CLANG_STYLE_IR), 9)

    def test_feature_extractor_uses_the_same_count(self):
        self.assertEqual(len(self.extractor.instructions), count_instructions(CLANG_STYLE_IR))

    def test_blocks(self):
        self.assertEqual(len(self.extractor.basic_blocks), 5)
        self.assertAlmostEqual(self.features["bb_density"], 5 / 9)

    def test_switch_edges(self):
        # switch: 3 targets; then three "br label %8"                 -> 6 edges
        self.assertAlmostEqual(self.features["cfg_edges_density"], 6 / 9)
        self.assertAlmostEqual(self.features["switch_density"], 1 / 9)
        # block 8 has 4 incoming edges, the entry block has 3 outgoing ones
        self.assertAlmostEqual(self.features["blocks_gt2_pred_density"], 1 / 9)
        self.assertAlmostEqual(self.features["blocks_gt2_succ_density"], 1 / 9)

    def test_bitwise_and_shift(self):
        self.assertAlmostEqual(self.features["bitwise_density"], 2 / 9)   # and, xor
        self.assertAlmostEqual(self.features["shift_density"], 1 / 9)     # shl

    def test_reachable_counts(self):
        # block 7 is dead: LLVM's analysis would report 8 instructions, 4 blocks
        self.assertEqual(reachable_counts(CLANG_STYLE_IR, "clang_style"), (8, 4))

    def test_constants(self):
        # 255 | 2 | switch cases 0 and 4 | -1 | phi 1 and 0
        self.assertAlmostEqual(self.features["const_operands_density"], 7 / 9)


class TestLoopInfoParsing(unittest.TestCase):
    """parse_loop_info on text copied from `opt -passes='print<loops>'`."""

    TEXT = (
        "Loop at depth 1 containing: %12<header><exiting>,%14,%15,%47,%48<latch>,%17,%22,%44,%45,%24,%42\n"
        "    Loop at depth 2 containing: %15<header><exiting>,%17,%22,%44,%45<latch>,%24,%42\n"
        "        Loop at depth 3 containing: %22<header><exiting>,%24,%42<latch>\n"
    )

    def test_parse(self):
        loops = parse_loop_info(self.TEXT)
        self.assertEqual(len(loops), 3)
        self.assertEqual([lp["depth"] for lp in loops], [1, 2, 3])
        self.assertEqual([lp["header"] for lp in loops], ["12", "15", "22"])
        self.assertEqual(loops[2]["blocks"], ["22", "24", "42"])
        self.assertEqual(len(loops[0]["blocks"]), 11)


class TestHashing(unittest.TestCase):
    """The hash must ignore comments, metadata and the function's own name."""

    A = "define i32 @f(i32 %0) #0 {\n  %2 = add i32 %0, 1, !dbg !7\n  ret i32 %2\n}\n"
    B = "; a comment\ndefine i32 @g(i32 %0) #3 {\n  %2 = add i32 %0, 1\n  ret i32 %2\n}\n"
    C = "define i32 @f(i32 %0) #0 {\n  %2 = add i32 %0, 2\n  ret i32 %2\n}\n"

    def test_same_body_same_hash(self):
        self.assertEqual(hash_ir(self.A, "f"), hash_ir(self.B, "g"))

    def test_different_body_different_hash(self):
        self.assertNotEqual(hash_ir(self.A, "f"), hash_ir(self.C, "f"))

    def test_normalize_removes_metadata(self):
        self.assertNotIn("!dbg", normalize_ir(self.A))
        self.assertNotIn("#0", normalize_ir(self.A))


@unittest.skipUnless(shutil.which("opt-17") or shutil.which("opt"), "LLVM opt not installed")
class TestAgainstRealOpt(unittest.TestCase):
    """Run the real opt on a fixture and check the loop features end to end."""

    def test_loop_sum_with_opt(self):
        from scripts.utils import find_llvm_tools
        from scripts.phase2_features import extract_features_from_ir_text
        tools = find_llvm_tools()
        ir_text = (FIXTURES_DIR / "loop_sum.ll").read_text()
        features = extract_features_from_ir_text(ir_text, tools, func_name="loop_sum")
        self.assertAlmostEqual(features["loop_count_density"], 1 / 10)
        self.assertEqual(features["max_loop_depth"], 1)
        self.assertEqual(features["max_loop_header_size"], 8)
        self.assertEqual(features["innermost_loop_body_size"], 8)


# ============================================================================
# Run tests standalone
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
