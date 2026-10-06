# Data Sources

## PolyBench/C 4.2.1
- **URL**: https://github.com/MatthiasJReis);  
  Fallback: https://downloads.sourceforge.net/project/polybench/polybench-c-4.2.1.tar.gz
- **Version**: 4.2.1
- **Commit / Tag**: v4.2.1 (if using GitHub)
- **Downloaded**: (auto-filled by phase1_setup_data.py)
- **Description**: 30 numerical computation kernels from linear algebra, stencils,
  data mining, and medley categories. Canonical compiler benchmark suite.

## MiBench
- **URL**: https://github.com/embecosm/mibench
- **Commit**: master (pinned at download time; hash recorded below)
- **Downloaded**: (auto-filled by phase1_setup_data.py)
- **Description**: Embedded benchmark suite with automotive, consumer, network,
  security, and telecom workloads. ~35 programs.

## AnghaBench (capped, optional)
- **URL**: https://github.com/brenocfg/AnghaBench
- **Commit**: master (if used, first 500 files from `linux/` subdirectory)
- **Downloaded**: (auto-filled if needed by phase1_setup_data.py)
- **Description**: 1 million compilable C functions from GitHub. Used only if
  PolyBench + MiBench yield < 1000 unique functions after deduplication.

---
*This file is auto-updated by `scripts/phase1_setup_data.py`.*
