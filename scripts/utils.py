"""
utils.py — Shared utilities for the LLVM IR SHAP research pipeline.

Provides:
  - LLVM tool discovery and version checking
  - Subprocess helpers with timeouts
  - IR hashing and normalization
  - Instruction counting
  - Project path helpers
  - Logging setup
"""

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# ============================================================================
# Project layout
# ============================================================================

# All paths are relative to the project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"          # downloaded source tarballs
IR_DIR         = DATA_DIR / "ir"           # baseline .ll files (one per function)
FEATURES_DIR   = DATA_DIR / "features"     # feature CSVs
LABELS_DIR     = DATA_DIR / "labels"       # label CSVs
CACHE_DIR      = DATA_DIR / "cache"        # opt result cache (keyed by IR hash)
RESULTS_DIR    = PROJECT_ROOT / "results"
FIGURES_DIR    = RESULTS_DIR / "figures"

ALL_DIRS = [DATA_DIR, RAW_DIR, IR_DIR, FEATURES_DIR, LABELS_DIR,
            CACHE_DIR, RESULTS_DIR, FIGURES_DIR]


def ensure_dirs():
    """Create every project directory if it does not exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


MODELS_DIR = DATA_DIR / "models"


def check_disk_space(min_free_gb: float = 1.0):
    """
    Check free space on the project drive. If below min_free_gb, raise and stop.
    """
    import shutil as _shutil
    total, used, free = _shutil.disk_usage(str(PROJECT_ROOT))
    free_gb = free / (1024 ** 3)
    log.info("Disk space: %.1f GB free (%.1f GB used)", free_gb, used / (1024 ** 3))
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Disk space critically low: {free_gb:.1f} GB free < {min_free_gb} GB minimum. "
            f"Stopping to prevent data loss."
        )



# ============================================================================
# Logging
# ============================================================================

def setup_logging(name: str = "llvm_shap", level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stderr with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = setup_logging()


# ============================================================================
# LLVM tool discovery
# ============================================================================

# Minimum LLVM version we support
MIN_LLVM_VERSION = 17

# Tool names (may have version suffixes like clang-17)
_TOOL_NAMES = ["clang", "opt", "llvm-extract", "llvm-dis", "llvm-as"]


def _find_tool(name: str) -> Optional[str]:
    """Find an LLVM tool, trying bare name first, then with version suffixes."""
    # Try bare name
    path = shutil.which(name)
    if path:
        return path
    # Try with version suffixes 17-20
    for ver in range(20, MIN_LLVM_VERSION - 1, -1):
        path = shutil.which(f"{name}-{ver}")
        if path:
            return path
            
    # Try via WSL if on Windows
    if sys.platform == "win32":
        try:
            res = subprocess.run(["wsl", "which", name], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                return f"wsl:{name}"
            for ver in range(20, MIN_LLVM_VERSION - 1, -1):
                res = subprocess.run(["wsl", "which", f"{name}-{ver}"], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    return f"wsl:{name}-{ver}"
        except Exception:
            pass
            
    return None


def find_llvm_tools() -> dict:
    """
    Locate all required LLVM tools.
    Returns a dict mapping tool name -> absolute path.
    Raises RuntimeError if any tool is missing.
    """
    tools = {}
    missing = []
    for name in _TOOL_NAMES:
        path = _find_tool(name)
        if path is None:
            missing.append(name)
        else:
            tools[name] = path
    if missing:
        raise RuntimeError(
            f"Missing LLVM tools: {missing}. "
            f"Install LLVM >= {MIN_LLVM_VERSION} or run setup_wsl.sh."
        )
    return tools


def get_llvm_version(tools: dict) -> str:
    """Extract the LLVM version string from clang --version."""
    cmd = tools["clang"]
    if sys.platform == "win32" and isinstance(cmd, str) and cmd.startswith("wsl:"):
        full_cmd = ["wsl", "--exec", cmd.split(":")[1], "--version"]
    else:
        full_cmd = [cmd, "--version"]
        
    result = subprocess.run(
        full_cmd,
        capture_output=True, text=True, timeout=10,
    )
    # Parse something like "clang version 17.0.6"
    match = re.search(r"clang version (\d+\.\d+\.\d+)", result.stdout)
    if match:
        return match.group(1)
    # Fallback: just grab the first version-like string
    match = re.search(r"(\d+\.\d+\.\d+)", result.stdout)
    return match.group(1) if match else "unknown"


# ============================================================================
# Subprocess helpers
# ============================================================================

def run_tool(
    cmd: list,
    timeout: int = 60,
    check: bool = True,
    input_data: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """
    Run a subprocess with a timeout.  Logs the command on failure.
    Returns the CompletedProcess object.
    """
    try:
        actual_cmd = []
        is_wsl = False
        if sys.platform == "win32" and isinstance(cmd[0], str) and cmd[0].startswith("wsl:"):
            is_wsl = True
            actual_cmd = ["wsl", "--exec", cmd[0].split(":")[1]]
            def wsl_path(m):
                drive = m.group(1).lower()
                rest = m.group(2).replace('\\', '/')
                return f"/mnt/{drive}/{rest}"
            for arg in cmd[1:]:
                arg_str = str(arg)
                arg_str = re.sub(r'^([a-zA-Z]):[\\/](.*)$', wsl_path, arg_str)
                arg_str = arg_str.replace('\\', '/')
                actual_cmd.append(arg_str)
        else:
            actual_cmd = [str(x) for x in cmd]

        result = subprocess.run(
            actual_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=input_data,
        )
        if check and result.returncode != 0:
            log.error("Command failed: %s", " ".join(cmd))
            log.error("stderr: %s", result.stderr[:2000])
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        return result
    except subprocess.TimeoutExpired:
        log.warning("Timeout (%ds) for: %s", timeout, " ".join(cmd))
        raise


# ============================================================================
# IR utilities
# ============================================================================

def normalize_ir(ir_text: str) -> str:
    """
    Normalize LLVM IR text for deduplication.
    Strips comments, metadata IDs, and debug locations so that
    semantically identical functions hash the same.
    """
    lines = []
    for line in ir_text.splitlines():
        # Remove comments
        line = re.sub(r";.*$", "", line)
        # Remove metadata references like !dbg !42
        line = re.sub(r"!dbg !\d+", "", line)
        # Remove metadata definitions like !42 = ...
        if re.match(r"^!\d+\s*=", line):
            continue
        # Normalize whitespace
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def hash_ir(ir_text: str) -> str:
    """SHA256 hash of normalized IR text, for deduplication and caching."""
    normalized = normalize_ir(ir_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def count_instructions(ir_text: str) -> int:
    """
    Count the number of LLVM IR instructions in the text.
    An instruction is any non-label, non-comment, non-empty line inside
    a function body that is not a metadata definition.
    """
    count = 0
    in_function = False
    for line in ir_text.splitlines():
        stripped = line.strip()
        # Skip empty lines and comments
        if not stripped or stripped.startswith(";"):
            continue
        # Track function boundaries
        if stripped.startswith("define "):
            in_function = True
            continue
        if stripped == "}" and in_function:
            in_function = False
            continue
        if not in_function:
            continue
        # Skip labels (lines ending with ':' that aren't instructions)
        if re.match(r"^[a-zA-Z_.][a-zA-Z0-9_.]*:\s*(;.*)?$", stripped):
            continue
        # Skip metadata definitions
        if re.match(r"^!\d+\s*=", stripped):
            continue
        # This is an instruction
        count += 1
    return count


def read_ir_file(path: Path) -> str:
    """Read an .ll file and return its text content."""
    return path.read_text(encoding="utf-8", errors="replace")


def write_ir_file(path: Path, content: str):
    """Write IR text to an .ll file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ============================================================================
# Environment info
# ============================================================================

def collect_env_info(tools: dict, seeds: list) -> dict:
    """Collect environment information for reproducibility (env.json)."""
    import importlib
    packages = {}
    for pkg_name in ["xgboost", "shap", "sklearn", "pandas", "matplotlib",
                     "numpy", "scipy", "yaml", "joblib", "tqdm", "seaborn"]:
        try:
            mod = importlib.import_module(pkg_name)
            packages[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[pkg_name] = "not installed"

    # Get git commit hash if available
    git_hash = "N/A"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except Exception:
        pass

    return {
        "llvm_version": get_llvm_version(tools),
        "llvm_tools": {k: v for k, v in tools.items()},
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "packages": packages,
        "seeds": seeds,
        "git_commit": git_hash,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def save_env_json(tools: dict, seeds: list):
    """Write env.json to the project root."""
    info = collect_env_info(tools, seeds)
    env_path = PROJECT_ROOT / "env.json"
    with open(env_path, "w") as f:
        json.dump(info, f, indent=2)
    log.info("Saved %s", env_path)


# ============================================================================
# Pass list management
# ============================================================================

# The 18 intraprocedural passes we study.
# Names are for the LLVM new pass manager (LLVM >= 17).
PASS_LIST = [
    "sroa",
    "instcombine",
    "simplifycfg",
    "early-cse",
    "gvn",
    "sccp",
    "adce",
    "dse",
    "reassociate",
    "jump-threading",
    "correlated-propagation",
    "licm",
    "loop-rotate",
    "indvars",
    "loop-deletion",
    "loop-idiom",
    "loop-unroll",
    "tailcallelim",
]


def verify_passes(tools: dict) -> list:
    """
    Verify each pass name against `opt --print-passes`.
    Returns a list of invalid pass names (empty if all OK).
    """
    result = run_tool([tools["opt"], "--print-passes"], check=False, timeout=10)
    # opt --print-passes outputs all known pass names to stdout
    known_text = result.stdout + result.stderr
    invalid = []
    for p in PASS_LIST:
        # Check if the pass name appears in the output
        # The output lists passes line by line, sometimes with descriptions
        if p not in known_text:
            log.warning("Pass '%s' not found in opt --print-passes output", p)
            invalid.append(p)
    return invalid


# ============================================================================
# Smoke-test helpers
# ============================================================================

# For --smoke mode, only process these programs (fast, small)
SMOKE_PROGRAMS = [
    "2mm",       # PolyBench
    "atax",      # PolyBench
    "bicg",      # PolyBench
]
