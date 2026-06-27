# Installation

This document describes recommended installation methods.

---

## 1. Recommended method with conda

```bash
cd STRAND
conda env create -f environment.yml
conda activate subcellular
pip install -e . --no-deps
```

Check installation:

```bash
subcellfeat --help
subcellfeat-pattern --help
subcellfeat-compartment --help
subcellfeat-coloc --help
```

---

## 2. Pip-only method

If your network is available and system dependencies are compatible:

```bash
cd STRAND
pip install -e .
```

This will install the toolbox and dependencies listed in `pyproject.toml`.

---

## 3. Developer installation

If dependencies are already installed and you only want to register the package and CLI commands:

```bash
pip install -e . --no-deps --no-build-isolation
```

---

## 4. Verify active environment

Before installing or running, check:

```bash
which python
which pip
```
---

