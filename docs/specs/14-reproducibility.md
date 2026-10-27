# Reproducibility Contract

## 16. Reproducibility

Every published result must be reproducible via a complete experiment definition.

At a minimum, the following are required:

```text
Experiment ID
Game Configuration
Bet Abstraction
Bot Version
Opponent Version
Opponent Parameters
Population Distribution
Population Version
Master Seed
RNG Configuration
Engine Version
Python Version
Dependency Lockfile (Poetry)
Poetry Lock Hash
Git Commit
CPU Model
CPU Core Count
CPU Architecture
OS Identifier
Sample Size
Power Target
Precision Target
Statistical Method
Calibration Results
```

An experiment receives a unique configuration hash. The hash is a SHA-256 digest of a canonical, sorted, normalised JSON representation of the frozen configuration.

Python dependencies are locked with **Poetry** using `pyproject.toml` and `poetry.lock`. The lockfile state is stored with every experiment configuration.

A result is considered reproducible only if a second run with identical configuration produces the same raw data or deterministically identical results.

For non‑deterministic extensions, the reproducibility of the defined statistical distribution and the experimental design is documented instead.

---
