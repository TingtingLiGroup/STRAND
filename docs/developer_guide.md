# Developer Guide

This document describes how to maintain and extend STRAND Tools.

---

## 1. Project structure

Recommended structure:

```text
Subcellular_ToolBox/
├── README.md
├── pyproject.toml
├── environment.yml
├── requirements.txt
├── LICENSE
├── tools/
├── models/
├── docs/
└── tests/
```

Do not include large input datasets, generated outputs, logs, or cache files in the released package.

---

## 2. Main source folders

```text
tools/cli/       command-line entry points
tools/api/       high-level Python APIs
tools/engines/   backend adapters for Bento, SPRAWL, compartment, and colocalization
tools/io/        input/output utilities
tools/models/    model wrappers
tools/utils/     shared utility functions
```

---

## 3. CLI entry points

Defined in `pyproject.toml`:

```toml
[project.scripts]
subcellfeat = "tools.cli.run:main"
subcellfeat-pattern = "tools.cli.pattern:main"
subcellfeat-compartment = "tools.cli.compartments:main"
subcellfeat-coloc = "tools.cli.colocalization:main"
```

Every CLI file should expose:

```python
def main():
    ...
```

---

## 4. Editable installation

For development:

```bash
conda activate subcellular
pip install -e . --no-deps --no-build-isolation
```

After editing source code, reinstalling is usually not necessary in editable mode, but reinstall when changing `pyproject.toml` or CLI entry points.

---

## 5. Dependency management

Keep dependency versions consistent across:

```text
pyproject.toml
requirements.txt
environment.yml
```

Important version constraints:

```text
numpy >=1.24,<2.0
matplotlib >=3.6,<3.10
scipy >=1.10,<1.14
shapely >=2.0,<2.2
geopandas >=0.13,<1.1
```

Avoid unbounded dependencies such as:

```text
numpy
matplotlib
scipy
```

because pip may install incompatible newest versions.

---

## 6. Adding a new CLI command

1. Create a new file under `tools/cli/`, for example:

```text
tools/cli/new_command.py
```

2. Add a `main()` function.

3. Register the command in `pyproject.toml`:

```toml
[project.scripts]
subcellfeat-new-command = "tools.cli.new_command:main"
```

4. Reinstall:

```bash
pip install -e . --no-deps --no-build-isolation
```

5. Test:

```bash
subcellfeat-new-command --help
```

---

## 7. Model management

Official pattern models are stored in `models/`:

```text
multiclass_xgb_8class_prop075_final_from_cv.joblib
multiclass_xgb_7class_no_foci_final_from_cv.joblib
```

If replacing a model, make sure the joblib object contains:

```text
model
feature_cols
classes
```

If feature columns or classes change, update:

```text
tools/models/pattern_classifier.py
docs/pattern_module.md
README.md
```

---

## 8. Testing

Run minimal tests:

```bash
pytest tests/
```

At minimum, tests should check:

1. `import tools`
2. `subcellfeat --help`
3. `subcellfeat-pattern --help`
4. `subcellfeat-compartment --help`
5. `subcellfeat-coloc --help`

---

## 9. Release checklist

Before packaging or handing off:

```bash
find . -type d -name "__pycache__"
find . -type f -name "*.pyc"
find . -type d -name "__MACOSX"
find . -type f -name ".DS_Store"
```

These commands should return no output.

Confirm commands:

```bash
subcellfeat --help
subcellfeat-pattern --help
subcellfeat-compartment --help
subcellfeat-coloc --help
```

Confirm required files:

```text
README.md
pyproject.toml
environment.yml
requirements.txt
LICENSE
models/
docs/
tests/
```
