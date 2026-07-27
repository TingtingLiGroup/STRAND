# Changelog

## Unreleased

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
