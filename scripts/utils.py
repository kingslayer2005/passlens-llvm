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

# The LLVM major version this project is pinned to. Thresholds in
# heuristics.yaml were read from the release/17.x sources, so labels must be
# generated with the same release.
PINNED_LLVM_VERSION = 17

# Tool names (may have version suffixes like clang-17)
_TOOL_NAMES = ["clang", "opt", "llvm-extract", "llvm-dis", "llvm-as"]


def _find_tool(name: str) -> Optional[str]:
    """
    Find an LLVM tool.

    Order of preference:
      1. The pinned version suffix, e.g. "clang-17" (PINNED_LLVM_VERSION).
         This matters on machines where the bare name "clang" points to a
         different LLVM release than the one the project is pinned to.
      2. The bare name, e.g. "clang".
      3. Other version suffixes (newest first).
      4. On Windows only: the same names inside WSL (slow bridge, see README).
    """
    # 1. Pinned version first, so labels are reproducible across machines
    path = shutil.which(f"{name}-{PINNED_LLVM_VERSION}")
    if path:
        return path
    # 2. Bare name
    path = shutil.which(name)
    if path:
        return path
    # 3. Any other supported version suffix
    for ver in range(20, MIN_LLVM_VERSION - 1, -1):
        path = shutil.which(f"{name}-{ver}")
        if path:
            return path

    # 4. Try via WSL if on Windows
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


def require_pinned_llvm(tools: dict) -> str:
    """
    Stop the run unless the LLVM tools are the pinned major version.
    Labels depend on the exact behaviour of each pass, so mixing LLVM
    releases would silently change the dataset.
    Set PASSLENS_ALLOW_ANY_LLVM=1 to override (the version is still logged).
    Returns the version string.
    """
    version = get_llvm_version(tools)
    major = version.split(".")[0]
    if major != str(PINNED_LLVM_VERSION):
        message = (f"LLVM {version} found, but the project is pinned to "
                   f"LLVM {PINNED_LLVM_VERSION}.x (tools: {tools}).")
        if os.environ.get("PASSLENS_ALLOW_ANY_LLVM") == "1":
            log.warning("%s Continuing because PASSLENS_ALLOW_ANY_LLVM=1.", message)
        else:
            raise RuntimeError(message + " Install LLVM 17 or set PASSLENS_ALLOW_ANY_LLVM=1.")
    return version


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

        # encoding/errors are set explicitly: compiler messages can quote
        # source lines that are not valid UTF-8, and that must not crash us.
        result = subprocess.run(
            actual_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            input=input_data,
        )
        if check and result.returncode != 0:
            log.error("Command failed: %s", " ".join(str(c) for c in cmd))
            log.error("stderr: %s", result.stderr[:2000])
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        return result
    except subprocess.TimeoutExpired:
        log.warning("Timeout (%ds) for: %s", timeout, " ".join(str(c) for c in cmd))
        raise


# ============================================================================
# IR utilities
# ============================================================================

# A basic-block label line.  LLVM prints a label as either
#     name:        where name uses the characters  - a-z A-Z $ . _ 0-9
#     "any text":  (quoted, when the name has other characters)
# optionally followed by a comment such as "; preds = %3, %7".
# NOTE: clang release builds discard value names, so most labels are purely
# numeric ("12:").  The label pattern MUST accept those.
_LABEL_RE = re.compile(r'^(?:([-a-zA-Z$._0-9]+)|"((?:[^"\\]|\\.)*)"):\s*(;.*)?$')

# Lines that belong to the PREVIOUS instruction (they are not instructions):
#   - the second line of an invoke / callbr:   "to label %a unwind label %b"
#   - landingpad clauses:                       "catch ...", "cleanup", "filter ..."
_CONTINUATION_PREFIXES = ("to label ", "catch ", "cleanup", "filter ")


def iter_function_body(ir_text: str):
    """
    Walk the body of every function DEFINITION in an IR module and yield one
    tuple per meaningful line:

        ("label", block_name)        a basic-block label
        ("inst",  instruction_text)  one instruction

    This is the ONE place that decides what counts as an instruction.  Both
    the label generator (count_instructions) and the feature extractor use
    it, so the two can never disagree.

    The count matches LLVM's own TotalInstructionCount (checked in Phase 1
    against `opt -passes='print<func-properties>'`).
    """
    in_function = False    # are we between "define ... {" and "}" ?
    pending_switch = None  # text of a switch whose case list is still open

    for line in ir_text.splitlines():
        stripped = line.strip()

        # Skip empty lines and whole-line comments
        if not stripped or stripped.startswith(";"):
            continue

        # Function start
        if not in_function:
            if stripped.startswith("define "):
                in_function = True
                pending_switch = None
            continue

        # Inside a multi-line switch: every line up to "]" is a case, not an
        # instruction.  We glue the cases onto the switch text so that the
        # caller sees ONE instruction that lists every branch target:
        #     switch i32 %x, label %default [
        #       i32 0, label %a
        #       i32 1, label %b
        #     ]
        if pending_switch is not None:
            pending_switch += " " + stripped
            if stripped.startswith("]"):
                yield ("inst", pending_switch)
                pending_switch = None
            continue

        # Function end
        if stripped == "}":
            in_function = False
            continue

        # Basic-block label
        label_match = _LABEL_RE.match(stripped)
        if label_match:
            name = label_match.group(1) if label_match.group(1) is not None else label_match.group(2)
            yield ("label", name)
            continue

        # Continuation lines of invoke / landingpad
        if stripped.startswith(_CONTINUATION_PREFIXES):
            continue

        # A switch whose case list continues on the following lines ends
        # with "[" (ignore a trailing comment when checking).  Hold it back
        # until the closing "]" arrives.
        code_part = stripped.split(";")[0].rstrip()
        if code_part.endswith("["):
            pending_switch = code_part
            continue

        # Anything else inside a function body is one instruction
        yield ("inst", stripped)


def count_instructions(ir_text: str) -> int:
    """
    Count LLVM IR instructions in all function definitions of the module.
    Labels, comments, switch case lines and metadata are NOT instructions.
    """
    count = 0
    for kind, _ in iter_function_body(ir_text):
        if kind == "inst":
            count += 1
    return count


# Metadata attachment at the end of an instruction, e.g. ", !dbg !42" or
# ", !llvm.loop !7" or ", !tbaa !3"
_METADATA_ATTACHMENT_RE = re.compile(r",?\s*![a-zA-Z_.][a-zA-Z0-9_.]*\s+!\d+")
# Attribute-group reference, e.g. "#0"
_ATTR_GROUP_RE = re.compile(r"\s#\d+")


def normalize_ir(ir_text: str) -> str:
    """
    Normalize LLVM IR text so that two modules that differ only in comments,
    metadata or attribute-group numbering compare equal.

    Removed: comments, metadata attachments (!dbg, !llvm.loop, !tbaa ...),
    metadata definitions (!42 = ...), named metadata, attribute groups,
    the ModuleID / source_filename lines, and extra whitespace.
    """
    lines = []
    for line in ir_text.splitlines():
        # Remove comments (text after ';').  String constants in C benchmarks
        # can contain ';' but they live in global initializers, which do not
        # affect the function-body comparison we use this for.
        line = re.sub(r";.*$", "", line)
        stripped = line.strip()
        if not stripped:
            continue
        # Drop metadata definitions like "!42 = ..." and "!llvm.loop = ..."
        if stripped.startswith("!"):
            continue
        # Drop attribute group definitions and the source file name
        if stripped.startswith("attributes #") or stripped.startswith("source_filename"):
            continue
        # Remove metadata attachments and attribute group references
        stripped = _METADATA_ATTACHMENT_RE.sub("", stripped)
        stripped = _ATTR_GROUP_RE.sub("", stripped)
        # Collapse runs of whitespace
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def function_body_text(ir_text: str) -> str:
    """
    Return the normalized text of the function DEFINITIONS only
    (from each "define" line to its closing "}").
    Declarations, globals, types and target lines are left out.
    """
    out = []
    in_function = False
    for line in normalize_ir(ir_text).splitlines():
        if not in_function and line.startswith("define "):
            in_function = True
        if in_function:
            out.append(line)
            if line == "}":
                in_function = False
    return "\n".join(out)


def hash_ir(ir_text: str, func_name: Optional[str] = None) -> str:
    """
    Hash of the normalized function body, used for deduplication, caching
    and the train/test leakage audit.

    Only the function definition is hashed (not the surrounding module), so
    the same function copied into two source files gets the same hash.
    If func_name is given, the function's own name is replaced by a fixed
    placeholder, so two identical functions with different names also match.
    """
    body = function_body_text(ir_text)
    if func_name:
        # Replace "@name" when it is not followed by another name character
        body = re.sub(r"@" + re.escape(func_name) + r"(?![-a-zA-Z$._0-9])", "@__FUNC__", body)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


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
