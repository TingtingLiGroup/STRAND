# Changelog

## Unreleased

- Upgrade the default RNA localization pattern classifier to a scale-invariant 28-feature architecture (`v2_28d` + `models/multiclass_xgb_8class_v2_28d_sparse_aug.joblib`, stored in cross-version portable XGBoost JSON booster format):
  - Per-cell centroid centering and span normalization (`cell_span -> 100.0`) unify pixel- and micron-coordinate datasets without unit drift, eliminating the Ripley `H(r)` scale mismatch while preserving empty-compartment `NaN` semantics.
  - Add 6 same-cell continuous CSR null Z-score features (`csr_z_nuclear_frac`, `csr_z_cell_edge_dist`, `csr_z_perinuc_cyto_dist`, `csr_z_centroid_shift`, `csr_z_nn_clustering`, `csr_z_rayleigh_radial`) and 5 cell-morphology/abundance covariates (`feat_log1p_n_spots`, `feat_nuclear_fraction`, `feat_cell_nc_ratio`, `feat_cell_aspect_ratio`, `feat_cell_solidity`), trained with `[6, 20]` sparse transcript downsampling augmentation.
  - Retain full backward compatibility: `subcellfeat --feature-set v1_17d` switches back to the legacy 17-feature pipeline and model (`multiclass_xgb_8class_prop075_final_from_cv.joblib`), and `subcellfeat-pattern` automatically detects 28-feature versus legacy 17-feature input tables.
- Report the evidence behind every localization call: pattern outputs now carry `pattern_prob` (posterior probability of the assigned class), `n_transcripts` (transcripts the call was computed from) and `low_abundance` (`n_transcripts` below `--low-abundance-cut`, default 10). Feature tables produced from a PKL bundle carry `n_transcripts` as well, so `subcellfeat-pattern` can propagate it; feature tables generated before this change still work and simply omit the two count-derived columns.
- Document how `pattern_prob` maps onto held-out accuracy, and why a low transcript count is not by itself a low-quality call (`docs/pattern_module.md` §9).
- Guard `subcellfeat --fast` so it fails early unless `--features-only` is set; fast mode omits two SPRAWL features and cannot feed the 17-feature pattern classifier.
- Include optimized module test directories in the default pytest configuration.
- Update README/CHANGELOG wording to match the current public command list and primary-only 8-class pattern default.

## 0.1.0 (2026-06-27)

Initial public release.

- RNA localization pattern classification (8-class XGBoost; historical no-Foci fallback retained for explicit API compatibility)
- Subcellular compartment detection (SOM-based, GPU optional)
- RNA colocalization analysis (InSTAnT-based PP + CPB)
- Standardized PKL input format
- Public CLI commands for pattern classification, compartment detection, compartment redisplay, colocalization, colocalization plotting, and colocalization summary
