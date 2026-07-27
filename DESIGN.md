# STRAND Tools Design

## Design Goals

STRAND Tools packages the analysis code needed to run subcellular spatial transcriptomics workflows from one standardized PKL bundle. The package exposes command-line and Python API entry points for three core tasks: RNA localization pattern classification, subcellular compartment detection, and RNA-RNA colocalization analysis.

The main goals are:

- Keep the public workflow simple: one input bundle, explicit output paths, and module-specific commands.
- Preserve reproducible defaults for released analyses while keeping compatibility options visible.
- Reuse proven internal engines for Bento/SPRAWL features, RNAflux/SOM compartments, and InSTAnT PP/CPB colocalization instead of reimplementing core statistics in each CLI.
- Keep large-data behavior practical by providing optimized backends where the original implementation is too memory hungry.

## Package Structure

- `tools/cli/`: command-line entry points and argument parsing.
- `tools/api/`: stable Python-facing orchestration functions.
- `tools/engines/`: adapters and workflow engines for feature, compartment, and visualization tasks.
- `tools/instant_optimized/`: low-memory InSTAnT-compatible colocalization backend.
- `tools/compartment_optimized/`: low-memory same-cell RNAflux/SOM compartment backend.
- `tools/models/` and `models/`: pattern-classifier loading code and packaged model files.
- `tests/` and module-local `tests/`: import, CLI, compatibility, and optimized-backend equivalence checks.

## Key Decisions

### Standard PKL Bundle

The package uses a single dictionary-style PKL contract because current project datasets already share this shape. The required transcript table is `data_df` with `cell`, `gene`, `x`, and `y`; cell and nuclear boundaries are stored as dictionaries keyed by cell ID. This keeps command-line use direct and avoids forcing users to convert every dataset into a heavier object model.

### Primary-Only Pattern Default

The public CLI defaults to the primary 8-class XGBoost model and preserves `Foci` predictions. The historical 7-class no-Foci model remains available only for explicit compatibility use through API parameters. This avoids hidden model switching based on the predicted Foci fraction.

### Fast Feature Mode

`subcellfeat --fast` computes only the two lightweight SPRAWL features, `sprawl_peripheral` and `sprawl_central`. Because the packaged pattern classifiers expect all 17 features, fast mode is limited to `--features-only`. Users who need pattern prediction should keep all four SPRAWL features and reduce approximation parameters instead.

### Optimized Backends

The optimized colocalization backend keeps InSTAnT PP/CPB semantics while reducing memory usage with sparse or chunked representations. The optimized compartment backend keeps same-cell RNAflux/SOM behavior while using cell indexing, memmapped arrays, and chunked output. These backends are exposed as explicit modules or default adapters where compatibility has been tested.

## Tradeoffs

- PKL input is simple and matches existing data, but it cannot be streamed before loading.
- CLI functions expose many parameters because the underlying scientific workflows are parameterized; this favors transparent command-line reproducibility over a small function signature.
- Optimized backends add implementation complexity, but they are isolated and covered by equivalence tests.
- Some integration tests depend on large LFS data and are disabled by default; small CLI and equivalence tests remain part of the default local test suite.

## Known Limits

- Full pattern prediction requires all 17 feature columns.
- Large PKL files still need enough memory for the initial load.
- Some plotting commands import Matplotlib at startup, so locked or unwritable user config directories may require setting `MPLCONFIGDIR`.
- Data integration tests require `git lfs pull` and `STRAND_RUN_DATA_TESTS=1`.

## Change History

- 2026-07-27: Documented the optimized STRAND release structure, primary-only 8-class pattern default, fast-feature constraint, and expanded default test coverage.
