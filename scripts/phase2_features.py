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
    iter_function_body,
)

log = setup_logging("phase2")

# ============================================================================
# Opcode classification
# ============================================================================

# Integer arithmetic opcodes
INT_ARITH_OPS = {"add", "sub", "mul", "udiv", "sdiv", "urem", "srem"}

# Bitwise logic opcodes (and / or / xor)
BITWISE_OPS = {"and", "or", "xor"}

# Shift opcodes
SHIFT_OPS = {"shl", "lshr", "ashr"}

# Floating-point arithmetic opcodes ("fneg" is the unary FP negate)
FP_ARITH_OPS = {"fadd", "fsub", "fmul", "fdiv", "frem", "fneg"}

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
        """
        Parse the IR text into basic blocks and instructions.

        Uses utils.iter_function_body, the same line classifier that
        count_instructions uses, so len(self.instructions) is always equal
        to count_instructions(ir_text).
        """
        # The entry block has no printed label in clang output, so we give
        # it a fixed internal name.  Nothing can branch to the entry block,
        # so this name never has to match a branch target.
        current_bb_label = "<entry>"
        current_bb_insts = []
        seen_any_line = False

        for kind, text in iter_function_body(self.ir_text):
            if kind == "label":
                # A label starts a new block.  Save the previous block,
                # unless it is the empty implicit entry block that exists
                # only because the first real block had an explicit label.
                if current_bb_insts or seen_any_line:
                    self.basic_blocks.append((current_bb_label, current_bb_insts))
                current_bb_label = text
                current_bb_insts = []
                seen_any_line = True
                continue

            # kind == "inst"
            seen_any_line = True
            opcode = self._extract_opcode(text)
            self.instructions.append((opcode, text))
            current_bb_insts.append((opcode, text))

            # Count phi arguments
            if opcode == "phi":
                self.phi_args_total += self._count_phi_args(text)

            # Classify calls
            if opcode in ("call", "invoke"):
                self._classify_call(text)

            # Count constant operands
            self.const_operands += self._count_constants(text)

            # Track CFG edges from terminators
            if opcode in ("br", "switch", "invoke", "indirectbr"):
                self._extract_cfg_edges(current_bb_label, opcode, text)

        # Save the last basic block
        if current_bb_insts:
            self.basic_blocks.append((current_bb_label, current_bb_insts))

    def _extract_opcode(self, line: str) -> str:
        """
        Extract the opcode from an instruction line.

        Patterns:
          %result = opcode ...
          opcode ...           (for void instructions like store, br, ret)
          tail call ...        (tail calls)
          musttail call ...

        Every instruction gets an opcode.  Opcodes we do not put in a
        feature class are still returned (and still counted as
        instructions); they simply do not add to any class density.
        """
        # Remove leading assignment: "%foo = ", "%12 = " or '%"odd name" = '
        inst_part = line
        assign_match = re.match(r'^%(?:[-a-zA-Z$._0-9]+|"(?:[^"\\]|\\.)*")\s*=\s*(.+)$', line)
        if assign_match:
            inst_part = assign_match.group(1)

        # The first word of inst_part is the opcode (or a call modifier)
        words = inst_part.split()
        if not words:
            return "unknown"

        opcode = words[0]

        # Handle modifiers like 'tail call', 'musttail call', 'notail call'
        if opcode in ("tail", "musttail", "notail") and len(words) > 1:
            opcode = words[1]

        return opcode

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

    # --- patterns used by _count_constants ---
    # Text that contains digits but is NOT an operand: alignments, array and
    # vector lengths, and numeric attributes.
    _NOT_OPERAND_RE = re.compile(
        r"\balign\s+\d+"                                  # align 8
        r"|\[\d+\s+x\s"                                   # [900 x double]
        r"|<(?:vscale\s+x\s+)?\d+\s+x\s"                  # <4 x float>
        r"|\b(?:align|dereferenceable|dereferenceable_or_null|addrspace|allocsize)\([^)]*\)"
    )
    # A numeric literal that stands on its own.  The look-behind rejects
    # digits that are part of a name or type: %12, @f1, i32, !7, #0, %.03
    _NUMBER_RE = re.compile(
        r"(?<![\w.%@!#$\"-])"
        r"(?:-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|0x[0-9A-Fa-f]+)"
        r"(?![\w.])"
    )
    _CONST_KEYWORD_RE = re.compile(r"\b(?:true|false|null|undef|poison|zeroinitializer)\b")

    def _count_constants(self, line: str) -> int:
        """
        Count constant operands in an instruction: integer and floating-point
        literals, and the keywords true / false / null / undef / poison /
        zeroinitializer.

        Examples:
          add nsw i32 %i, 1                        -> 1   (the literal 1)
          %p = phi i32 [ 0, %entry ], [ %n, %b ]   -> 1   (the literal 0)
          store double 0.000000e+00, ptr %21, align 8  -> 1  (align is not an operand)
          getelementptr [900 x double], ptr %6, i64 0, i64 %20 -> 1
        """
        # Drop the "%result = " part so a numeric result name is never counted
        text = re.sub(r'^%(?:[-a-zA-Z$._0-9]+|"(?:[^"\\]|\\.)*")\s*=\s*', "", line)
        # Drop alignments, array lengths and numeric attributes
        text = self._NOT_OPERAND_RE.sub(" ", text)
        return len(self._NUMBER_RE.findall(text)) + len(self._CONST_KEYWORD_RE.findall(text))

    def _extract_cfg_edges(self, src_label: str, opcode: str, line: str):
        """Extract CFG edges from terminator instructions."""
        # Branch targets look like:  label %name   or   label %"odd name"
        # (utils.iter_function_body joins a multi-line switch into one text,
        #  so every case target is on this line too)
        for plain, quoted in re.findall(r'label\s+%(?:([-a-zA-Z$._0-9]+)|"((?:[^"\\]|\\.)*)")', line):
            self.cfg_edges.append((src_label, plain if plain else quoted))

    # ================================================================
    # Feature computation
    # ================================================================

    def extract_features(self, loops: Optional[List[dict]] = None) -> Dict[str, float]:
        """
        Compute all features and return as a dict.

        loops: the list returned by parse_loop_info() for this same IR
               (one dict per loop with its depth and member blocks).
               If None, the loop features are left out; callers that need
               them must pass it (see extract_features_from_ir_text).

        Only size-independent features are returned: densities
        (count / instruction_count), per-block and per-loop sizes, the
        maximum loop depth, and log_inst_count.  Raw counts are NOT returned.
        """
        features = {}
        inst_count = len(self.instructions)

        if inst_count == 0:
            log.warning("Function %s has 0 instructions", self.func_name)
            return features

        # -- Opcode class counts --
        opcode_counter = Counter(op for op, _ in self.instructions)

        counts = {}

        # Integer arithmetic, bitwise logic and shifts
        counts["int_arith_count"] = sum(opcode_counter.get(op, 0) for op in INT_ARITH_OPS)
        counts["bitwise_count"] = sum(opcode_counter.get(op, 0) for op in BITWISE_OPS)
        counts["shift_count"] = sum(opcode_counter.get(op, 0) for op in SHIFT_OPS)

        # FP arithmetic
        counts["fp_arith_count"] = sum(opcode_counter.get(op, 0) for op in FP_ARITH_OPS)

        # Comparisons
        counts["icmp_count"] = opcode_counter.get("icmp", 0)
        counts["fcmp_count"] = opcode_counter.get("fcmp", 0)

        # Memory
        counts["load_count"] = opcode_counter.get("load", 0)
        counts["store_count"] = opcode_counter.get("store", 0)
        counts["gep_count"] = opcode_counter.get("getelementptr", 0)
        counts["alloca_count"] = opcode_counter.get("alloca", 0)

        # SSA
        counts["phi_count"] = opcode_counter.get("phi", 0)

        # Calls (total)
        counts["call_count"] = opcode_counter.get("call", 0) + opcode_counter.get("invoke", 0)

        # Casts
        counts["cast_count"] = sum(opcode_counter.get(op, 0) for op in CAST_OPS)

        # Control flow
        counts["select_count"] = opcode_counter.get("select", 0)
        counts["ret_count"] = opcode_counter.get("ret", 0)
        counts["invoke_count"] = opcode_counter.get("invoke", 0)
        counts["switch_count"] = opcode_counter.get("switch", 0)

        # Branches: conditional ("br i1 %c, label %a, label %b") versus
        # unconditional ("br label %a")
        cond_br = 0
        uncond_br = 0
        for op, line in self.instructions:
            if op == "br":
                if line.startswith("br i1 "):
                    cond_br += 1
                else:
                    uncond_br += 1
        counts["cond_br_count"] = cond_br
        counts["uncond_br_count"] = uncond_br

        # -- Structural features --
        counts["bb_count"] = len(self.basic_blocks)

        # CFG edges
        counts["cfg_edges"] = len(self.cfg_edges)

        # Predecessor / successor distributions
        pred_count = Counter()  # label -> number of incoming edges
        succ_count = Counter()  # label -> number of outgoing edges
        for src, dst in self.cfg_edges:
            pred_count[dst] += 1
            succ_count[src] += 1

        # Blocks by predecessor count
        all_labels = [label for label, _ in self.basic_blocks]
        counts["blocks_1_pred"] = sum(1 for l in all_labels if pred_count.get(l, 0) == 1)
        counts["blocks_2_pred"] = sum(1 for l in all_labels if pred_count.get(l, 0) == 2)
        counts["blocks_gt2_pred"] = sum(1 for l in all_labels if pred_count.get(l, 0) > 2)

        # Blocks by successor count
        counts["blocks_1_succ"] = sum(1 for l in all_labels if succ_count.get(l, 0) == 1)
        counts["blocks_2_succ"] = sum(1 for l in all_labels if succ_count.get(l, 0) == 2)
        counts["blocks_gt2_succ"] = sum(1 for l in all_labels if succ_count.get(l, 0) > 2)

        # Phi arguments total
        counts["phi_args_total"] = self.phi_args_total

        # Constant operands
        counts["const_operands"] = self.const_operands

        # Call classification
        counts["direct_call_count"] = self.direct_calls
        counts["intrinsic_call_count"] = self.intrinsic_calls
        counts["self_recursive_call_count"] = self.self_recursive_calls

        # -- Density features (count / instruction_count) --
        # "xxx_count" becomes "xxx_density"; names without "_count" get
        # "_density" appended.  Every density therefore ends in "_density".
        for count_name, value in counts.items():
            if count_name.endswith("_count"):
                density_name = count_name[: -len("_count")] + "_density"
            else:
                density_name = count_name + "_density"
            features[density_name] = value / inst_count

        # -- Size features (not divided by the instruction count) --
        features["log_inst_count"] = math.log(inst_count + 1)

        # Mean and max instructions per basic block
        bb_sizes = {label: len(insts) for label, insts in self.basic_blocks}
        features["mean_insts_per_bb"] = inst_count / max(len(bb_sizes), 1)
        features["max_insts_per_bb"] = float(max(bb_sizes.values())) if bb_sizes else 0.0

        # -- Loop features, from LLVM's own LoopInfo (exact, not guessed) --
        if loops is not None:
            features.update(self._loop_features(loops, bb_sizes, inst_count))

        return features

    def _loop_features(self, loops: List[dict], bb_sizes: Dict[str, int],
                       inst_count: int) -> Dict[str, float]:
        """
        Turn LoopInfo's list of loops into features.

          loop_count_density       number of loops / instruction count
          max_loop_depth           deepest nesting level (0 = no loops)
          max_loop_header_size     instructions in the largest loop header
          min_loop_header_size     instructions in the smallest loop header
          innermost_loop_body_size instructions in the smallest innermost
                                   loop (all of its blocks added together)

        All sizes are 0 when the function has no loops.
        A loop is "innermost" when no other loop's header is among its blocks.
        """
        out = {}
        out["loop_count_density"] = len(loops) / inst_count
        out["max_loop_depth"] = float(max((lp["depth"] for lp in loops), default=0))

        header_sizes = []
        innermost_body_sizes = []
        all_headers = set(lp["header"] for lp in loops)

        for lp in loops:
            # Every block LoopInfo names must exist in our parse; if not, the
            # two parsers disagree and the loop sizes would be wrong.
            missing = [b for b in lp["blocks"] if b not in bb_sizes]
            if missing:
                raise ValueError(
                    f"LoopInfo names blocks that the IR parser did not find "
                    f"in {self.func_name}: {missing[:5]}")

            header_sizes.append(bb_sizes[lp["header"]])

            # Innermost loop: contains no other loop's header
            other_headers_inside = (set(lp["blocks"]) & all_headers) - {lp["header"]}
            if not other_headers_inside:
                innermost_body_sizes.append(sum(bb_sizes[b] for b in lp["blocks"]))

        out["max_loop_header_size"] = float(max(header_sizes, default=0))
        out["min_loop_header_size"] = float(min(header_sizes, default=0))
        out["innermost_loop_body_size"] = float(min(innermost_body_sizes, default=0))
        return out


def reachable_counts(ir_text: str, func_name: Optional[str] = None) -> Tuple[int, int]:
    """
    Count the instructions and basic blocks that are reachable from the
    entry block, using OUR parse of the blocks and branch targets.

    LLVM's own function-properties analysis counts reachable blocks only,
    so this is the number to compare against LLVM when validating the
    parser.  (Labels use the total count, which also includes dead blocks.)
    Returns (reachable_instructions, reachable_blocks).
    """
    extractor = IRFeatureExtractor(ir_text, func_name)
    if not extractor.basic_blocks:
        return 0, 0
    sizes = {label: len(insts) for label, insts in extractor.basic_blocks}
    successors = defaultdict(set)
    for src, dst in extractor.cfg_edges:
        successors[src].add(dst)

    # Walk the control-flow graph from the entry block (the first block)
    entry = extractor.basic_blocks[0][0]
    visited = set([entry])
    work = [entry]
    while work:
        block = work.pop()
        for nxt in successors[block]:
            if nxt not in visited and nxt in sizes:
                visited.add(nxt)
                work.append(nxt)
    return sum(sizes[b] for b in visited), len(visited)


# ============================================================================
# Loop analysis using opt
# ============================================================================

def parse_loop_info(loop_output: str) -> List[dict]:
    """
    Parse the text printed by `opt -passes='print<loops>'`.

    Each loop is one line, for example:
        Loop at depth 2 containing: %15<header><exiting>,%17,%22,%45<latch>
    Returns a list of dicts: {"depth": 2, "header": "15", "blocks": ["15", ...]}
    """
    loops = []
    for line in loop_output.splitlines():
        match = re.search(r"Loop at depth (\d+) containing: (.*)$", line)
        if not match:
            continue
        depth = int(match.group(1))
        blocks = []
        header = None
        for item in match.group(2).split(","):
            item = item.strip()
            if not item:
                continue
            # Split "%15<header><exiting>" into the name and its tags
            name = item.split("<")[0].strip()
            if name.startswith("%"):
                name = name[1:]
            # Quoted names are printed as %"name"
            if len(name) >= 2 and name.startswith('"') and name.endswith('"'):
                name = name[1:-1]
            blocks.append(name)
            if "<header>" in item:
                header = name
        if header is None and blocks:
            header = blocks[0]   # LoopInfo always prints the header first
        loops.append({"depth": depth, "header": header, "blocks": blocks})
    return loops


def run_noop_with_loops(ir_text: str, tools: dict, timeout: int = 10) -> Tuple[str, List[dict]]:
    """
    Run opt once with only the loop printer.  One process gives us both:
      - the IR re-printed by opt with NO transformation (the "no-op" text
        that pass outputs are compared against to decide "fired")
      - LoopInfo for the loop features
    Raises CalledProcessError / TimeoutExpired on failure (never guesses).
    """
    cmd = [tools["opt"], "-passes=print<loops>", "-S"]
    result = run_tool(cmd, timeout=timeout, check=True, input_data=ir_text)
    return result.stdout, parse_loop_info(result.stderr)


def get_loop_info(ll_file: Optional[Path], tools: dict, ir_text: Optional[str] = None) -> Tuple[int, int]:
    """
    Loop count and maximum loop depth for a file or IR text.
    Kept for callers that only need these two numbers.
    Raises on failure instead of returning a made-up (0, 0).
    """
    if ir_text is None:
        if ll_file is None or not ll_file.exists():
            raise FileNotFoundError(f"No IR given for loop analysis: {ll_file}")
        ir_text = read_ir_file(ll_file)
    _, loops = run_noop_with_loops(ir_text, tools)
    return len(loops), max((lp["depth"] for lp in loops), default=0)


# ============================================================================
# Main pipeline
# ============================================================================

def extract_features_for_file(ll_file: Path, tools: dict,
                               func_name: Optional[str] = None) -> Optional[Dict]:
    """Extract features from a single .ll file."""
    ir_text = read_ir_file(ll_file)
    features = extract_features_from_ir_text(ir_text, tools, func_name=func_name)
    return features if features else None


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
                                    func_name: Optional[str] = None,
                                    loops: Optional[List[dict]] = None) -> Dict:
    """
    Extract the full feature vector from IR text.

    loops: LoopInfo already parsed for this IR (saves one opt call).
           If None, opt is run here to get it.
    """
    if loops is None:
        _, loops = run_noop_with_loops(ir_text, tools)
    extractor = IRFeatureExtractor(ir_text, func_name)
    return extractor.extract_features(loops=loops)


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
