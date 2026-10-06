import shutil
import os
import subprocess
from pathlib import Path

def get_free_space_gb(drive="C:\\"):
    return shutil.disk_usage(drive).free / (1024**3)

print(f"C_FREE_BEFORE_GB={get_free_space_gb():.2f}")

# 1. pip cache purge
print("Purging Windows pip cache...")
subprocess.run(["pip", "cache", "purge"])

print("Purging WSL pip cache...")
# Assuming .venv is in ~ or somewhere, but let's just use pip3 or python3 -m pip inside wsl
subprocess.run(["wsl", "--exec", "bash", "-c", "pip3 cache purge || ~/.venv/bin/pip cache purge"])

# 2. apt clean and autoremove (Requires password, skipping)
print("Skipping apt clean in WSL (requires sudo)...")
print("Skipping apt autoremove in WSL (requires sudo)...")

# 3. Clean project files
proj = Path(".")
print("Cleaning regenerable project files...")

# __pycache__
for p in proj.rglob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)

# .ll and .bc in data/ir
ir_dir = proj / "data" / "ir"
if ir_dir.exists():
    for f in ir_dir.rglob("*.ll"):
        f.unlink()
    for f in ir_dir.rglob("*.bc"):
        f.unlink()

# archives in data/raw
raw_dir = proj / "data" / "raw"
if raw_dir.exists():
    for f in raw_dir.rglob("*.tar.gz"):
        f.unlink()
    for f in raw_dir.rglob("*.zip"):
        f.unlink()
    for f in raw_dir.rglob("*.tgz"):
        f.unlink()

# mibench data files (audio, images, large text)
mibench = proj / "data" / "raw" / "mibench"
if mibench.exists():
    for ext in ["*.dat", "*.pcm", "*.au", "*.pgm", "*.jpg", "*.wav", "*.txt", "*.bin"]:
        for f in mibench.rglob(ext):
            f.unlink()

# 4. Find ext4.vhdx
vhdx_paths = list(Path(os.environ["LOCALAPPDATA"]).joinpath("Packages").rglob("ext4.vhdx"))
print(f"VHDX Paths: {[str(p) for p in vhdx_paths]}")

print(f"C_FREE_AFTER_GB={get_free_space_gb():.2f}")
