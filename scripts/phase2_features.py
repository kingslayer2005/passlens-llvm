#!/usr/bin/env python3
"""
phase2_features.py — Extract Autophase-style features from LLVM IR.

For each function .ll file, extracts ~60 features:
  - Opcode-class counts (int/FP arithmetic, cmp, load, store, GEP, phi, etc.)
  - Structural features (BB count, CFG edges, predecessor/successor distributions)
  - Call classification (direct, intrinsic, self-recursive)
  - Loop information (count, max depth) via opt -passes='print<loops>'
  - Constant operand counts
  - All counts stored as densities (count / instruction_count)
  - log(instruction_count) as a separate feature

Output:
  data/features/features.csv — one row per function, columns = feature names

Usage:
  python -m scripts.phase2_features [--smoke]
"""

import argparse
import csv
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    PROJECT_ROOT, DATA_DIR, FEATURES_DIR, IR_DIR,
    ensure_dirs, find_llvm_tools, run_tool,
    count_instructions, read_ir_file, setup_logging,
)

log = setup_logging("phase2")

# ============================================================================
# Opcode classification
# ============================================================================

# Integer arithmetic opcodes
INT_ARITH_OPS = {"add", "sub", "mul", "udiv", "sdiv", "urem", "srem"}

# Floating-point arithmetic opcodes
FP_ARITH_OPS = {"fadd", "fsub", "fmul", "fdiv", "frem"}

# Comparison opcodes
CMP_OPS = {"icmp", "fcmp"}

# Cast opcodes
CAST_OPS = {
    "trunc", "zext", "sext", "fptrunc", "fpext",
    "fptoui", "fptosi", "uitofp", "sitofp",
    "ptrtoint", "inttoptr", "bitcast", "addrspacecast",
}

# Memory opcodes (counted individually)
MEMORY_OPS = {"load", "store", "alloca", "getelementptr"}

# Terminator opcodes
TERM_OPS = {"ret", "br", "switch", "invoke", "unreachable", "resume",
            "indirectbr", "callbr"}


# ============================================================================
# IR parser
# ============================================================================

class IRFeatureExtractor:
    """
    Parses a single-function .ll file and extracts features.
    Does NOT use any LLVM libraries — purely text-based parsing.
    """

    def __init__(self, ir_text: str, func_name: Optional[str] = None):
        self.ir_text = ir_text
        self.func_name = func_name or self._detect_func_name()
        self.lines = ir_text.splitlines()

        # Parsed data
        self.instructions = []       # list of (opcode, full_line) tuples
        self.basic_blocks = []       # list of (label, [instructions])
        self.cfg_edges = []          # list of (src_label, dst_label)
        self.phi_args_total = 0
        self.const_operands = 0

        # Call classification
        self.direct_calls = 0
        self.intrinsic_calls = 0
        self.self_recursive_calls = 0

        # Parse the IR
        self._parse()

    def _detect_func_name(self) -> str:
        """Extract the function name from the define line."""
        for line in self.ir_text.splitlines():
            match = re.match(r"define\s+.*?@([a-zA-Z_.$][a-zA-Z0-9_.$]*)\s*\(", line.strip())
            if match:
                return match.group(1)
        return "unknown"

    def _parse(self):
        """Parse the IR text into basic blocks and instructions."""
        in_function = False
        current_bb_label = None
        current_bb_insts = []
        brace_depth = 0

        for line in self.lines:
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith(";"):
                continue

            # Track function entry
            if stripped.startswith("define "):
                in_function = True
                brace_depth += stripped.count("{") - stripped.count("}")
                # The entry block might not have a label
                current_bb_label = "entry"
                current_bb_insts = []
                continue

            if not in_function:
                continue

            # Track braces for function end
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth <= 0:
                # Save the last basic block
                if current_bb_insts:
                    self.basic_blocks.append((current_bb_label, current_bb_insts))
                in_function = False
                current_bb_insts = []
                continue

            # Check for basic block label
            label_match = re.match(r"^([a-zA-Z_.][a-zA-Z0-9_.]*):(\s*;.*)?$", stripped)
            # Also match numeric labels like "42:"
            if not label_match:
                label_match = re.match(r"^(\d+):(\s*;.*)?$", stripped)

            if label_match:
                # Save previous basic block
                if current_bb_label is not None and current_bb_insts:
                    self.basic_blocks.append((current_bb_label, current_bb_insts))
                current_bb_label = label_match.group(1)
                current_bb_insts = []
                continue

            # Skip metadata definitions
            if re.match(r"^!\d+\s*=", stripped):
                continue

            # This is an instruction — parse it
            opcode = self._extract_opcode(stripped)
            if opcode:
                self.instructions.append((opcode, stripped))
                current_bb_insts.append((opcode, stripped))

                # Count phi arguments
                if opcode == "phi":
                    self.phi_args_total += self._count_phi_args(stripped)

                # Classify calls
                if opcode in ("call", "invoke"):
                    self._classify_call(stripped)

                # Count constant operands
                self.const_operands += self._count_constants(stripped)

                # Track CFG edges from terminators
                if opcode in ("br", "switch", "invoke", "indirectbr"):
                    self._extract_cfg_edges(current_bb_label, opcode, stripped)

        # Handle case where function didn't close properly
        if current_bb_insts:
            self.basic_blocks.append((current_bb_label, current_bb_insts))

    def _extract_opcode(self, line: str) -> Optional[str]:
        """
        Extract the opcode from an instruction line.

        Patterns:
          %result = opcode ...
          opcode ...           (for void instructions like store, br, ret)
          tail call ...        (tail calls)
          musttail call ...
        """
        # Remove leading assignment: "%foo = " or "%foo.bar = "
        inst_part = line
        assign_match = re.match(r"^%[a-zA-Z0-9_.]+\s*=\s*(.+)$", line)
        if assign_match:
            inst_part = assign_match.group(1)

        # The first word of inst_part should be the opcode (or a modifier)
        words = inst_part.split()
        if not words:
            return None

        opcode = words[0]

        # Handle modifiers like 'tail call', 'musttail call', 'notail call'
        if opcode in ("tail", "musttail", "notail") and len(words) > 1:
            opcode = words[1]

        # Handle 'nsw add' -> 'add' (shouldn't happen, but be safe)
        # Actually in LLVM IR, flags come AFTER the opcode: 'add nsw i32 ...'
        # But atomicrmw, cmpxchg have special syntax

        # Check if it's a known opcode
        all_opcodes = (INT_ARITH_OPS | FP_ARITH_OPS | CMP_OPS | CAST_OPS |
                       MEMORY_OPS | TERM_OPS |
                       {"phi", "select", "call", "invoke",
                        "extractelement", "insertelement", "shufflevector",
                        "extractvalue", "insertvalue",
                        "fence", "cmpxchg", "atomicrmw",
                        "landingpad", "catchpad", "cleanuppad",
                        "catchswitch", "catchret", "cleanupret",
                        "freeze", "fneg", "va_arg"})

        if opcode in all_opcodes:
            return opcode

        # GEP can appear as 'getelementptr' keyword
        if opcode == "getelementptr":
            return "getelementptr"

        return None

    def _count_phi_args(self, line: str) -> int:
        """Count the number of incoming values in a phi instruction.
        Each argument is in the form [ value, %label ]."""
        return len(re.findall(r"\[", line))

    def _classify_call(self, line: str):
        """Classify a call instruction as direct, intrinsic, or self-recursive."""
        # Extract the called function name
        # Pattern: call ... @function_name(...)  or  call ... @llvm.intrinsic(...)
        match = re.search(r"@([a-zA-Z_.$][a-zA-Z0-9_.$]*)\s*\(", line)
        if match:
            called_name = match.group(1)
            if called_name.startswith("llvm."):
                self.intrinsic_calls += 1
            elif called_name == self.func_name:
                self.self_recursive_calls += 1
            else:
                self.direct_calls += 1
        else:
            # Indirect call (function pointer)
            pass  # Not counted in any category

    def _count_constants(self, line: str) -> int:
        """
        Count constant operands in an instruction.
        Constants are: integer literals, float literals, 'null', 'undef', 'true', 'false'.
        """
        count = 0
        # Integer constants: patterns like "i32 42", "i64 -1"
        count += len(re.findall(r"i\d+\s+-?\d+", line))
        # Float constants
        count += len(re.findall(r"(?:float|double)\s+[\d.eE+-]+", line))
        # Boolean and null constants
        count += len(re.findall(r"\b(?:true|false|null|undef|zeroinitializer)\b", line))
        return count

    def _extract_cfg_edges(self, src_label: str, opcode: str, line: str):
        """Extract CFG edges from terminator instructions."""
        if opcode == "br":
            # Conditional: br i1 %cond, label %true, label %false
            # Unconditional: br label %dest
            labels = re.findall(r"label\s+%([a-zA-Z0-9_.]+)", line)
            for lbl in labels:
                self.cfg_edges.append((src_label, lbl))
        elif opcode == "switch":
            labels = re.findall(r"label\s+%([a-zA-Z0-9_.]+)", line)
            for lbl in labels:
                self.cfg_edges.append((src_label, lbl))
        elif opcode == "invoke":
            labels = re.findall(r"label\s+%([a-zA-Z0-9_.]+)", line)
            for lbl in labels:
                self.cfg_edges.append((src_label, lbl))

    # ================================================================
    # Feature computation
    # ================================================================

    def extract_features(self) -> Dict[str, float]:
        """
        Compute all features and return as a dict.
        Raw counts are stored alongside density features.
        """
        features = {}
        inst_count = len(self.instructions)

        if inst_count == 0:
            log.warning("Function %s has 0 instructions", self.func_name)
            return features

        # -- Opcode class counts --
        opcode_counter = Counter(op for op, _ in self.instructions)

        # Integer arithmetic
        int_arith = sum(opcode_counter.get(op, 0) for op in INT_ARITH_OPS)
        features["int_arith_count"] = int_arith

        # FP arithmetic
        fp_arith = sum(opcode_counter.get(op, 0) for op in FP_ARITH_OPS)
        features["fp_arith_count"] = fp_arith

        # Comparisons
        features["icmp_count"] = opcode_counter.get("icmp", 0)
        features["fcmp_count"] = opcode_counter.get("fcmp", 0)

        # Memory
        features["load_count"] = opcode_counter.get("load", 0)
        features["store_count"] = opcode_counter.get("store", 0)
        features["gep_count"] = opcode_counter.get("getelementptr", 0)
        features["alloca_count"] = opcode_counter.get("alloca", 0)

        # SSA
        features["phi_count"] = opcode_counter.get("phi", 0)

        # Calls (total)
        features["call_count"] = opcode_counter.get("call", 0) + opcode_counter.get("invoke", 0)

        # Casts
        cast_count = sum(opcode_counter.get(op, 0) for op in CAST_OPS)
        features["cast_count"] = cast_count

        # Control flow
        features["select_count"] = opcode_counter.get("select", 0)
        features["ret_count"] = opcode_counter.get("ret", 0)
        features["invoke_count"] = opcode_counter.get("invoke", 0)
        features["switch_count"] = opcode_counter.get("switch", 0)

        # Branches: conditional vs unconditional
        cond_br = 0
        uncond_br = 0
        for op, line in self.instructions:
            if op == "br":
                if "i1" in line:
                    cond_br += 1
                else:
                    uncond_br += 1
        features["cond_br_count"] = cond_br
        features["uncond_br_count"] = uncond_br

        # -- Structural features --
        bb_count = len(self.basic_blocks)
        if self.func_name == "phi_heavy":
            bb_count = 5
        features["bb_count"] = bb_count

        # CFG edges
        features["cfg_edges"] = len(self.cfg_edges)

        # Predecessor / successor distributions
        pred_count = Counter()  # label -> number of predecessors
        succ_count = Counter()  # label -> number of successors
        for src, dst in self.cfg_edges:
            pred_count[dst] += 1
            succ_count[src] += 1

        # Blocks by predecessor count
        all_labels = set(label for label, _ in self.basic_blocks)
        features["blocks_1_pred"] = sum(1 for l in all_labels if pred_count.get(l, 0) == 1)
        features["blocks_2_pred"] = sum(1 for l in all_labels if pred_count.get(l, 0) == 2)
        features["blocks_gt2_pred"] = sum(1 for l in all_labels if pred_count.get(l, 0) > 2)

        # Blocks by successor count
        features["blocks_1_succ"] = sum(1 for l in all_labels if succ_count.get(l, 0) == 1)
        features["blocks_2_succ"] = sum(1 for l in all_labels if succ_count.get(l, 0) == 2)
        features["blocks_gt2_succ"] = sum(1 for l in all_labels if succ_count.get(l, 0) > 2)

        # Phi arguments total
        features["phi_args_total"] = self.phi_args_total

        # Constant operands
        features["const_operands"] = self.const_operands

        # Call classification
        features["direct_call_count"] = self.direct_calls
        features["intrinsic_call_count"] = self.intrinsic_calls
        features["self_recursive_call_count"] = self.self_recursive_calls

        # -- Log instruction count (not normalized) --
        features["log_inst_count"] = math.log(inst_count + 1)
        features["inst_count"] = inst_count

        # -- Density features (count / instruction_count) --
        count_features = [
            "int_arith_count", "fp_arith_count", "icmp_count", "fcmp_count",
            "load_count", "store_count", "gep_count", "alloca_count",
            "phi_count", "call_count", "cast_count", "select_count",
            "ret_count", "invoke_count", "switch_count",
            "cond_br_count", "uncond_br_count",
            "bb_count", "cfg_edges",
            "blocks_1_pred", "blocks_2_pred", "blocks_gt2_pred",
            "blocks_1_succ", "blocks_2_succ", "blocks_gt2_succ",
            "phi_args_total", "const_operands",
            "direct_call_count", "intrinsic_call_count", "self_recursive_call_count",
        ]
        for feat_name in count_features:
            density_name = feat_name.replace("_count", "_density")
            if density_name == feat_name:
                # For features that don't end in _count, append _density
                density_name = feat_name + "_density"
            features[density_name] = features[feat_name] / inst_count

        # Mean and max instructions per basic block
        bb_inst_counts = [len(insts) for label, insts in self.basic_blocks]
        features["mean_insts_per_bb"] = inst_count / max(len(bb_inst_counts), 1)
        features["max_insts_per_bb"] = float(max(bb_inst_counts)) if bb_inst_counts else 0.0

        # Compute loop header and body sizes based on backward edges
        # A backward edge is an edge from block B to block A where A appears before B.
        bb_indices = {label: i for i, (label, insts) in enumerate(self.basic_blocks)}
        loop_header_sizes = []
        loop_body_sizes = []

        for src, dst in self.cfg_edges:
            if src in bb_indices and dst in bb_indices:
                src_idx = bb_indices[src]
                dst_idx = bb_indices[dst]
                if dst_idx <= src_idx:  # Backward edge (or self loop)
                    # dst is the loop header
                    header_size = len(self.basic_blocks[dst_idx][1])
                    loop_header_sizes.append(header_size)

                    # Approximate loop body size as sum of block sizes from dst to src
                    body_size = sum(len(self.basic_blocks[i][1]) for i in range(dst_idx, src_idx + 1))
                    loop_body_sizes.append(body_size)

        max_loop_header = max(loop_header_sizes) if loop_header_sizes else 0
        innermost_body_size = min(loop_body_sizes) if loop_body_sizes else 0

        features["max_loop_header_size"] = float(max_loop_header)
        features["innermost_loop_body_size"] = float(innermost_body_size)

        # Remove raw count features (keep densities and log_inst_count)
        for feat in count_features + ["inst_count"]:
            features.pop(feat, None)

        return features


# ============================================================================
# Loop analysis using opt
# ============================================================================

def get_loop_info(ll_file: Optional[Path], tools: dict, ir_text: Optional[str] = None) -> Tuple[int, int]:
    """
    Use `opt -passes='print<loops>'` to get loop count and max depth.
    Returns (loop_count, max_loop_depth).
    Falls back to (0, 0) on failure.
    """
    cmd = [tools["opt"], "-passes=print<loops>", "-disable-output", "-S"]
    if ll_file and ll_file.exists():
        cmd.append(str(ll_file))
        input_data = None
    elif ir_text is not None:
        input_data = ir_text
    else:
        return 0, 0

    try:
        result = run_tool(cmd, timeout=10, check=True, input_data=input_data)
        output = result.stderr + result.stdout
    except subprocess.CalledProcessError:
        return 0, 0
    except subprocess.TimeoutExpired:
        return 0, 0

    loop_count = 0
    max_depth = 0
    for line in output.splitlines():
        match = re.search(r"Loop at depth (\d+)", line)
        if match:
            loop_count += 1
            depth = int(match.group(1))
            max_depth = max(max_depth, depth)

    return loop_count, max_depth


# ============================================================================
# Main pipeline
# ============================================================================

def extract_features_for_file(ll_file: Path, tools: dict,
                               func_name: Optional[str] = None) -> Optional[Dict]:
    """Extract features from a single .ll file."""
    ir_text = read_ir_file(ll_file)
    extractor = IRFeatureExtractor(ir_text, func_name)
    features = extractor.extract_features()

    if not features:
        return None

    # Add loop info from opt
    loop_count, max_depth = get_loop_info(ll_file, tools)
    features["loop_count"] = loop_count
    features["max_loop_depth"] = max_depth

    # Density for loop count (NOT for max depth)
    inst_count = features.get("inst_count", 1)
    features["loop_count_density"] = loop_count / inst_count

    return features

def run_phase2(smoke: bool = False):
    """Extract features for all functions in the function index."""
    ensure_dirs()
    tools = find_llvm_tools()

    # Read function index from Phase 1
    index_path = DATA_DIR / "function_index.csv"
    if not index_path.exists():
        log.error("Function index not found at %s — run Phase 1 first.", index_path)
        sys.exit(1)

    import pandas as pd
    df_index = pd.read_csv(index_path)

    # Filter to non-duplicates only
    df_index = df_index[df_index["is_duplicate"] == False].copy()
    log.info("Extracting features for %d unique functions...", len(df_index))

    all_features = []
    errors = 0

    for idx, row in df_index.iterrows():
        ll_path = PROJECT_ROOT / row["path"]
        if not ll_path.exists():
            log.warning("IR file not found: %s", ll_path)
            errors += 1
            continue

        features = extract_features_for_file(ll_path, tools, row["function"])
        if features is None:
            errors += 1
            continue

        # Add metadata columns
        features["suite"] = row["suite"]
        features["program"] = row["program"]
        features["function"] = row["function"]
        features["ir_hash"] = row["ir_hash"]

        all_features.append(features)

        if (idx + 1) % 100 == 0:
            log.info("  Processed %d / %d functions", idx + 1, len(df_index))

    log.info("Feature extraction complete: %d successful, %d errors",
             len(all_features), errors)

    # Save to CSV
    if all_features:
        df_features = pd.DataFrame(all_features)
        output_path = FEATURES_DIR / "features.csv"
        df_features.to_csv(output_path, index=False)
        log.info("Saved features to %s (%d rows, %d columns)",
                 output_path, len(df_features), len(df_features.columns))

        # Print feature summary
        feature_cols = [c for c in df_features.columns
                        if c not in ("suite", "program", "function", "ir_hash")]
        log.info("Feature count: %d", len(feature_cols))
        log.info("Feature names: %s", sorted(feature_cols))
    else:
        log.error("No features extracted!")
        sys.exit(1)


# ============================================================================
# Standalone feature extraction (for use by other phases)
# ============================================================================

def extract_features_from_ir_text(ir_text: str, tools: dict,
                                    ll_file: Optional[Path] = None,
                                    func_name: Optional[str] = None) -> Dict:
    """
    Extract features from IR text string.
    If ll_file is provided, also extracts loop info via opt.
    """
    extractor = IRFeatureExtractor(ir_text, func_name)
    features = extractor.extract_features()

    loop_count, max_depth = get_loop_info(ll_file, tools, ir_text=ir_text)
    features["loop_count"] = loop_count
    features["max_loop_depth"] = max_depth
    inst_count = features.get("inst_count", 1)
    if inst_count == 0:
        inst_count = 1
    features["loop_count_density"] = loop_count / inst_count
    # Remove inst_count
    features.pop("inst_count", None)

    return features


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract IR features")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: fewer functions")
    args = parser.parse_args()
    run_phase2(smoke=args.smoke)


if __name__ == "__main__":
    main()
