#!/usr/bin/env python3
"""
analysis_common.py — Shared building blocks for Phases 4 to 7.

Everything that must be IDENTICAL across phases lives here, so it cannot
drift apart:

  - the settings of the pre-specified analysis (ANALYSIS_PLAN.md)
  - how the dataset is loaded and which columns are features
  - which (pass, target) pairs are kept (the 5 %-95 % rule)
  - the cross-validation splits (same seeds -> same folds in every phase)
  - how an XGBoost model is built and tuned
  - small statistics helpers (PR-AUC, Holm correction, program bootstrap)
"""

import json
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.targets import TARGETS, build_target
from scripts.utils import (
    PROJECT_ROOT, DATA_DIR, FEATURES_DIR, LABELS_DIR, CACHE_DIR, RESULTS_DIR,
    PASS_LIST, setup_logging,
)

log = setup_logging("analysis")

# Columns of features.csv / labels.csv that describe a sample but are NOT
# model inputs.
META_COLS = ["func_id", "suite", "program", "function", "ir_hash", "state"]

MASTER_TABLE = RESULTS_DIR / "master_table.csv"
TUNED_PARAMS = RESULTS_DIR / "phase4_tuned_params.json"

MODEL_NAMES = ["majority", "logistic_regression", "random_forest", "xgboost"]


# ============================================================================
# Settings (ANALYSIS_PLAN.md, Amendment 2)
# ============================================================================

@dataclass(frozen=True)
class Settings:
    """Every number that shapes the analysis.  FULL is the pre-specified run."""
    name: str = "full"
    keep_low: float = 0.05            # a target is kept when its positive rate
    keep_high: float = 0.95           # lies in [keep_low, keep_high]
    n_repeats: int = 3                # outer CV repeats
    n_outer_folds: int = 5            # outer CV folds per repeat
    n_inner_folds: int = 3            # inner folds for hyperparameter tuning
    n_tuning_configs: int = 6         # random configurations tried per outer fold
    n_bootstrap: int = 2000           # program-level bootstrap resamples
    n_permutations: int = 10000       # random rankings for the heuristic null
    alpha: float = 0.05               # significance level after Holm correction
    stability_tau_min: float = 0.6    # minimum SHAP rank stability to interpret
    min_valid_folds: int = 10         # fewer usable outer folds -> not interpreted
    require_wilcoxon: bool = True     # the fold-level test is part of the gate
    base_seed: int = 42
    background_size: int = 100        # background rows for interventional SHAP
    interaction_samples: int = 300    # held-out rows for interaction values
    n_random_draws: int = 5           # random feature sets (faithfulness, pruning)
    faithfulness_k: Tuple[int, ...] = (1, 3, 5, 10)
    pruning_k: Tuple[int, ...] = (5, 10, 15, 20)
    sequence_max_steps: int = 8       # Phase 7: longest pass sequence
    sequence_max_functions: int = 200 # Phase 7: held-out functions evaluated
    sequence_holdout_fraction: float = 0.2


FULL = Settings()

# Smoke settings only check that the code runs end to end in minutes.
# Numbers produced with them are NOT results.
SMOKE = replace(
    FULL, name="smoke", n_repeats=1, n_outer_folds=3, n_inner_folds=2,
    n_tuning_configs=2, n_bootstrap=200, n_permutations=1000,
    min_valid_folds=2, require_wilcoxon=False,   # 3 folds are too few for the test
    interaction_samples=100, n_random_draws=2,
    faithfulness_k=(1, 5), pruning_k=(5, 10),
    sequence_max_steps=4, sequence_max_functions=12,
)

# Passes used by --smoke when no --only list is given (two ordinary passes
# and one loop pass)
SMOKE_PASSES = ["simplifycfg", "early-cse", "loop-rotate"]

# XGBoost hyperparameter search space (ANALYSIS_PLAN.md, section 4)
XGB_SEARCH_SPACE = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}


def get_settings(smoke: bool) -> Settings:
    return SMOKE if smoke else FULL


# ============================================================================
# Machine resources
# ============================================================================

def safe_n_jobs(requested: int) -> int:
    """
    Number of parallel worker processes that is safe on this machine.

    Each worker needs its own copy of Python, numpy, scikit-learn and
    XGBoost in memory (roughly 1 GB with head-room), so the number of
    workers is limited by free memory as well as by CPU cores.  Asking for
    one worker per core on a 20-core laptop with 7 GB of free memory makes
    the machine swap and hang.
    """
    cores = os.cpu_count() or 1
    limit = max(1, min(requested, cores))
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available_gb = int(line.split()[1]) / (1024 ** 2)
                    limit = max(1, min(limit, int(available_gb // 2)))
                    break
    except OSError:
        pass   # not Linux: keep the CPU-based limit
    if limit < requested:
        log.info("Using %d parallel job(s) instead of %d (cores / free memory).",
                 limit, requested)
    return limit


def free_disk_gb() -> float:
    """Free space, in GB, on the drive that holds the project."""
    return shutil.disk_usage(str(PROJECT_ROOT)).free / (1024 ** 3)


class DiskGuard:
    """
    Stops a run that eats disk space.  Create it at the start of a phase and
    call check() after every unit of work.
    """

    def __init__(self, max_drop_gb: float = 1.0, min_free_gb: float = 1.0):
        self.start_free = free_disk_gb()
        self.max_drop_gb = max_drop_gb
        self.min_free_gb = min_free_gb

    def check(self):
        free = free_disk_gb()
        if free < self.min_free_gb:
            raise RuntimeError(f"Only {free:.1f} GB free on the project drive. Stopping.")
        if self.start_free - free > self.max_drop_gb:
            raise RuntimeError(
                f"Free disk space fell by {self.start_free - free:.1f} GB during this "
                f"phase (limit {self.max_drop_gb} GB). Stopping; find what grew.")


# ============================================================================
# Dataset
# ============================================================================

def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load features.csv (one row per state) and labels.csv (one row per
    state x pass) and check the properties the analysis relies on.
    """
    features_path = FEATURES_DIR / "features.csv"
    labels_path = LABELS_DIR / "labels.csv"
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError("features.csv / labels.csv not found. Run Phases 1-3 first.")

    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    if not features["ir_hash"].is_unique:
        raise ValueError("features.csv has duplicate ir_hash values.")
    missing = set(labels["ir_hash"]) - set(features["ir_hash"])
    if missing:
        raise ValueError(f"{len(missing)} label states have no feature row.")
    return features, labels


def feature_columns(features: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return (feature_names, dropped_constant_columns).
    A feature is every numeric column that is not metadata.  Columns with a
    single value in the whole dataset carry no information and are dropped
    (and reported).
    """
    candidates = [c for c in features.columns if c not in META_COLS]
    non_numeric = [c for c in candidates if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")
    if features[candidates].isna().any().any():
        raise ValueError("features.csv contains missing values.")
    constant = [c for c in candidates if features[c].nunique() <= 1]
    kept = [c for c in candidates if c not in constant]
    return kept, constant


@dataclass
class PairData:
    """Everything needed to model one (pass, target) pair."""
    pass_name: str
    target: str
    X: np.ndarray            # (n_samples, n_features) float32
    y: np.ndarray            # (n_samples,) 0/1
    groups: np.ndarray       # benchmark program of each sample (CV group)
    suites: np.ndarray       # benchmark suite of each sample
    hashes: np.ndarray       # ir_hash of each sample
    func_ids: np.ndarray     # function each state came from
    feature_names: List[str]
    n_errors: int            # label rows removed because opt failed


def build_pair_data(features: pd.DataFrame, labels: pd.DataFrame, pass_name: str,
                    target: str, threshold: float = 0.01,
                    feature_names: Optional[List[str]] = None) -> PairData:
    """Join the label of one (pass, target) with the feature matrix."""
    if feature_names is None:
        feature_names, _ = feature_columns(features)

    pass_df = labels[labels["pass_name"] == pass_name]
    clean_df, y, n_errors = build_target(pass_df, pass_name, target, threshold)

    # Bring the feature rows into the order of the label rows
    feat = features.set_index("ir_hash").loc[clean_df["ir_hash"].to_numpy()]
    return PairData(
        pass_name=pass_name,
        target=target,
        X=feat[feature_names].to_numpy(dtype=np.float32),
        y=y,
        groups=feat["program"].to_numpy(),
        suites=feat["suite"].to_numpy(),
        hashes=clean_df["ir_hash"].to_numpy(),
        func_ids=feat["func_id"].to_numpy(),
        feature_names=list(feature_names),
        n_errors=n_errors,
    )


def target_table(labels: pd.DataFrame, cfg: Settings,
                 passes: Optional[List[str]] = None) -> pd.DataFrame:
    """
    One row for EVERY pass x target with its positive rate and whether the
    keep rule lets it through.  Nothing is fitted for a dropped target.
    """
    rows = []
    for pass_name in (passes or PASS_LIST):
        pass_df = labels[labels["pass_name"] == pass_name]
        for target in TARGETS:
            _, y, n_errors = build_target(pass_df, pass_name, target)
            n = len(y)
            rate = float(y.mean()) if n else float("nan")
            if n == 0:
                status, reason = "dropped", "no usable rows"
            elif rate < cfg.keep_low:
                status, reason = "dropped", f"positive rate {rate:.1%} is below {cfg.keep_low:.0%}"
            elif rate > cfg.keep_high:
                status, reason = "dropped", f"positive rate {rate:.1%} is above {cfg.keep_high:.0%}"
            else:
                status, reason = "kept", ""
            rows.append({
                "pass_name": pass_name, "target": target, "n_samples": n,
                "n_errors": n_errors, "n_positive": int(y.sum()),
                "positive_rate": round(rate, 4), "keep_status": status,
                "keep_reason": reason,
            })
    return pd.DataFrame(rows)


def select_passes(smoke: bool, only: Optional[str]) -> List[str]:
    """Which passes a run covers: --only list, the smoke subset, or all 18."""
    if only:
        chosen = [p.strip() for p in only.split(",") if p.strip()]
        unknown = [p for p in chosen if p not in PASS_LIST]
        if unknown:
            raise ValueError(f"Unknown pass name(s): {unknown}")
        return chosen
    return list(SMOKE_PASSES) if smoke else list(PASS_LIST)


# ============================================================================
# Cross-validation
# ============================================================================

def outer_splits(y: np.ndarray, groups: np.ndarray, repeat: int,
                 cfg: Settings) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    The outer folds of one repeat: StratifiedGroupKFold grouped by benchmark
    program, shuffled with a seed that depends only on the repeat number.
    Because the seed is fixed, every phase that calls this gets the SAME
    folds for the same (pass, target).
    """
    from sklearn.model_selection import StratifiedGroupKFold
    n_splits = min(cfg.n_outer_folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Need at least 2 programs for cross-validation.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=cfg.base_seed + repeat)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def fold_is_usable(y_train: np.ndarray, y_test: np.ndarray) -> Tuple[bool, str]:
    """A fold needs both classes on both sides, or PR-AUC is undefined."""
    if len(np.unique(y_train)) < 2:
        return False, "training fold has a single class"
    if len(np.unique(y_test)) < 2:
        return False, "test fold has a single class"
    return True, ""


# ============================================================================
# Models
# ============================================================================

def candidate_configs(pass_name: str, target: str, cfg: Settings) -> List[dict]:
    """
    The random hyperparameter configurations tried for one (pass, target).
    Drawn once from XGB_SEARCH_SPACE with a seed derived from the pair name,
    so every outer fold of the pair chooses among the same candidates.
    """
    # A stable seed from the names (Python's hash() changes between runs)
    seed = cfg.base_seed + sum(ord(ch) for ch in f"{pass_name}|{target}")
    rng = np.random.RandomState(seed)
    configs = []
    seen = set()
    while len(configs) < cfg.n_tuning_configs:
        config = {name: rng.choice(values).item() for name, values in XGB_SEARCH_SPACE.items()}
        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs


def make_xgb(params: dict, y_train: np.ndarray, seed: int):
    """
    An XGBoost classifier with the given hyperparameters.
    scale_pos_weight = negatives / positives handles class imbalance.
    n_jobs=1 because parallelism happens across (pass, target) pairs.
    """
    from xgboost import XGBClassifier
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    return XGBClassifier(
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
        eval_metric="logloss",
        scale_pos_weight=n_neg / max(n_pos, 1),
        **params,
    )


def tune_xgb(X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
             configs: List[dict], cfg: Settings, seed: int) -> Tuple[dict, str]:
    """
    Inner cross-validation on the TRAINING fold only: return the candidate
    configuration with the best mean PR-AUC, and a note on how it was chosen.
    The test fold is never seen here.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    n_splits = min(cfg.n_inner_folds, len(np.unique(groups_train)))
    if n_splits < 2:
        return configs[0], "not tuned: fewer than 2 programs in the training fold"

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner = [(tr, va) for tr, va in splitter.split(X_train, y_train, groups_train)
             if fold_is_usable(y_train[tr], y_train[va])[0]]
    if not inner:
        return configs[0], "not tuned: no inner fold has both classes"

    best_config, best_score = None, -1.0
    for config in configs:
        scores = []
        for tr, va in inner:
            model = make_xgb(config, y_train[tr], seed)
            model.fit(X_train[tr], y_train[tr])
            scores.append(pr_auc(y_train[va], model.predict_proba(X_train[va])[:, 1]))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_config, best_score = config, mean_score
    return best_config, f"tuned on {len(inner)} inner fold(s)"


def load_tuned_params() -> dict:
    """The hyperparameters Phase 4 chose, per (pass, target) and outer fold."""
    if not TUNED_PARAMS.exists():
        raise FileNotFoundError(f"{TUNED_PARAMS} not found. Run Phase 4 first.")
    with open(TUNED_PARAMS, encoding="utf-8") as f:
        return json.load(f)


def pair_key(pass_name: str, target: str) -> str:
    return f"{pass_name}|{target}"


def fold_key(repeat: int, fold: int) -> str:
    return f"r{repeat}f{fold}"


def usable_folds(data: "PairData", tuned_pair: dict, cfg: Settings, repeats=None):
    """
    Yield the outer folds that Phase 4 scored for this pair, with the
    hyperparameters it chose there:
        (repeat, fold, train_idx, test_idx, params, seed)
    The splits are rebuilt from the same seeds, so they are the same folds.
    Folds that Phase 4 skipped (a single class) are skipped here too.
    """
    for repeat in (range(cfg.n_repeats) if repeats is None else repeats):
        for fold, (train_idx, test_idx) in enumerate(outer_splits(data.y, data.groups, repeat, cfg)):
            entry = tuned_pair["folds"].get(fold_key(repeat, fold))
            if entry is None:
                continue
            seed = cfg.base_seed + 1000 * repeat + fold
            yield repeat, fold, train_idx, test_idx, entry["params"], seed


def tree_shap(model, X: np.ndarray) -> np.ndarray:
    """
    Path-dependent TreeSHAP values, one row per sample and one column per
    feature, in log-odds units.

    XGBoost computes these itself (pred_contribs=True).  It is the same
    TreeSHAP algorithm that shap.TreeExplainer uses with
    feature_perturbation="tree_path_dependent"; using XGBoost's own
    implementation keeps the main results independent of the shap package
    version.  (tests/test_analysis.py checks the two agree.)
    """
    import xgboost
    contributions = model.get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)
    return contributions[:, :-1]   # the last column is the bias term


def importance_share(shap_values: np.ndarray) -> np.ndarray:
    """
    Mean absolute SHAP value per feature, scaled to sum to 1.
    Scaling lets folds be averaged without one fold dominating.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    total = mean_abs.sum()
    return mean_abs / total if total > 0 else mean_abs


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Rank 1 = largest score.  Ties are broken by position (stable)."""
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


# ============================================================================
# Statistics helpers
# ============================================================================

def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Area under the precision-recall curve (average precision):
        sum over thresholds of (recall_k - recall_{k-1}) * precision_k
    Same definition as sklearn.metrics.average_precision_score, written out
    with numpy because the bootstrap calls it tens of thousands of times.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    n_pos = y_true.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    s_sorted = scores[order]
    # One threshold per distinct score: the last position of each tie group
    last_of_group = np.r_[np.where(np.diff(s_sorted))[0], len(s_sorted) - 1]
    true_pos = np.cumsum(y_sorted)[last_of_group]
    predicted_pos = last_of_group + 1
    precision = true_pos / predicted_pos
    recall = true_pos / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def holm_correction(p_values: List[float]) -> List[float]:
    """
    Holm-Bonferroni adjusted p-values.  NaN entries stay NaN and do not
    count as tests.
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    m = len(valid)
    if m == 0:
        return adjusted.tolist()
    order = valid[np.argsort(p[valid])]
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, value)   # keep the sequence monotone
        adjusted[idx] = running_max
    return adjusted.tolist()


def program_bootstrap_indices(groups: np.ndarray, n_bootstrap: int, seed: int):
    """
    Yield index arrays for a bootstrap that resamples whole PROGRAMS with
    replacement.  Samples of one program are correlated, so the program is
    the independent unit; resampling single rows would give intervals that
    are too narrow.
    """
    rng = np.random.RandomState(seed)
    programs = np.unique(groups)
    rows_of = {p: np.where(groups == p)[0] for p in programs}
    for _ in range(n_bootstrap):
        drawn = rng.choice(programs, size=len(programs), replace=True)
        yield np.concatenate([rows_of[p] for p in drawn])


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile confidence interval, ignoring NaN resamples."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    return (float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2))))


# ============================================================================
# Small file helpers
# ============================================================================

_FINGERPRINTS = {}


def run_fingerprint(cfg: Settings) -> str:
    """
    A short id of everything a cached result depends on: the two data files
    and the settings.  If the data or a setting changes, the id changes and
    old cache files are simply not found, so stale numbers cannot be reused.
    """
    if cfg.name not in _FINGERPRINTS:
        import hashlib
        digest = hashlib.sha256()
        for path in (FEATURES_DIR / "features.csv", LABELS_DIR / "labels.csv"):
            digest.update(path.read_bytes())
        digest.update(repr(cfg).encode("utf-8"))
        digest.update(json.dumps(XGB_SEARCH_SPACE, sort_keys=True).encode("utf-8"))
        _FINGERPRINTS[cfg.name] = digest.hexdigest()[:10]
    return _FINGERPRINTS[cfg.name]


def cache_dir(phase: str, cfg: Settings) -> Path:
    """Folder for the resume cache of one phase, tied to the data and settings."""
    path = CACHE_DIR / f"{phase}_{cfg.name}_{run_fingerprint(cfg)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_small(path: Path, payload: dict, max_mb: float = 1.0):
    """
    Write a cache file and refuse to write a big one.  A cache holds numbers
    for resuming a run; if it is megabytes large, something that does not
    belong there (a model, a DataFrame, raw SHAP values) slipped in.
    """
    text = json.dumps(payload)
    size_mb = len(text.encode("utf-8")) / (1024 ** 2)
    if size_mb > max_mb:
        raise RuntimeError(f"Refusing to write {path.name}: {size_mb:.1f} MB (limit {max_mb} MB).")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)   # atomic: an interrupted run never leaves half a file


def read_master_table() -> pd.DataFrame:
    if not MASTER_TABLE.exists():
        raise FileNotFoundError(f"{MASTER_TABLE} not found. Run Phase 4 first.")
    return pd.read_csv(MASTER_TABLE, keep_default_na=False, na_values=[""])
