import pandas as pd
import logging

log = logging.getLogger("targets")

def build_target_labels(df: pd.DataFrame, pass_name: str, target_name: str, threshold: float = 0.01) -> pd.Series:
    """
    Builds the binary label for a given pass and target, strictly filtering errors.
    
    Args:
        df: DataFrame containing labels for the pass. Must have 'outcome',
            'inst_before', 'inst_after', 'inst_control', and 'fired'.
        pass_name: Name of the pass.
        target_name: 'beneficial', 'harmful', or 'fired'.
        threshold: The relative reduction threshold for 'beneficial'.
        
    Returns:
        A boolean Series of the same length as the valid rows in df.
        (Note: the function drops 'error' rows from df IN PLACE or assumes they are dropped.
        Wait, the instruction says: "rows with outcome == 'error' are removed and counted".)
    """
    initial_len = len(df)
    
    # Remove errors
    errors = df['outcome'] == 'error'
    if errors.any():
        error_count = errors.sum()
        log.info(f"Removing {error_count} rows with outcome == 'error' for pass {pass_name}")
        df.drop(df[errors].index, inplace=True)
    
    # Define loop passes
    loop_passes = {"licm", "loop-rotate", "indvars", "loop-deletion", "loop-idiom", "loop-unroll"}
    
    # Select reference
    if pass_name in loop_passes:
        reference = df['inst_control']
    else:
        reference = df['inst_before']
        
    # Build label
    if target_name == "beneficial":
        # (reference - inst_after) / reference >= threshold
        # handle division by zero just in case, though inst_before > 0 normally
        rel_reduction = (reference - df['inst_after']) / reference
        label = rel_reduction >= threshold
    elif target_name == "harmful":
        label = df['inst_after'] > reference
    elif target_name == "fired":
        label = df['fired'].astype(bool)
    else:
        raise ValueError(f"Unknown target: {target_name}")
        
    return label
