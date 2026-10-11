#!/usr/bin/env python3
"""
targets.py — The ONE place that turns measured pass outcomes into labels.

Every phase that needs a label (Phases 4, 5, 6 and 7) calls build_target().
No other file may compute a label from the counts.

Three targets per pass:

  beneficial  the pass made the function at least `threshold` smaller
              (reference - inst_after) / reference >= threshold
  harmful     the pass made the function larger
              inst_after > reference
  fired       the pass changed the IR at all (compared with its control run)

`reference` is the instruction count the pass is compared against:

  - for the six loop passes it is inst_control, the count after LLVM's
    loop canonicalization (LoopSimplify + LCSSA) alone.  LLVM always runs
    that step before a loop pass, and it can add instructions on its own;
    comparing against it isolates what the pass itself did.
  - for every other pass it is inst_before, the count of the input state.

Rows whose opt call failed (outcome == "error") are removed and counted;
they are never treated as "no change".
"""

from typing import Tuple

import numpy as np
import pandas as pd

# The passes that LLVM wraps in a loop adaptor (see scripts/phase3_labels.py)
LOOP_PASSES = frozenset([
    "licm", "loop-rotate", "indvars", "loop-deletion", "loop-idiom", "loop-unroll",
])

TARGETS = ("beneficial", "harmful", "fired")

DEFAULT_THRESHOLD = 0.01   # 1 % relative reduction


def reference_count(pass_df: pd.DataFrame, pass_name: str) -> pd.Series:
    """The instruction count that `pass_name` is compared against (see above)."""
    if pass_name in LOOP_PASSES:
        return pass_df["inst_control"]
    return pass_df["inst_before"]


def build_target(pass_df: pd.DataFrame, pass_name: str, target: str,
                 threshold: float = DEFAULT_THRESHOLD) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Build the 0/1 label vector for one (pass, target) pair.

    pass_df : the label rows of ONE pass (one row per state).
    Returns : (clean_df, y, n_errors)
        clean_df  pass_df without the error rows (a copy; the input is not changed)
        y         numpy array of 0/1, one entry per row of clean_df
        n_errors  how many rows were removed because the opt call failed
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown target '{target}'. Expected one of {TARGETS}.")
    wrong_pass = pass_df["pass_name"] != pass_name
    if wrong_pass.any():
        raise ValueError(f"pass_df holds rows of other passes than '{pass_name}'.")

    # Remove failed opt calls and count them
    is_error = pass_df["outcome"] == "error"
    n_errors = int(is_error.sum())
    clean_df = pass_df.loc[~is_error].copy()

    reference = reference_count(clean_df, pass_name).astype(float)
    inst_after = clean_df["inst_after"].astype(float)

    if target == "beneficial":
        y = (reference - inst_after) / reference >= threshold
    elif target == "harmful":
        y = inst_after > reference
    else:  # "fired"
        y = clean_df["fired"].astype(int) == 1

    return clean_df, y.to_numpy().astype(int), n_errors
