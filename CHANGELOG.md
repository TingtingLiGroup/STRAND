# Changelog

## Unreleased

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
