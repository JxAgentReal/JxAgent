# Packaging Validation

Date: 2026-08-19

This public package was derived from the latest available `JxOS_10of10_regression_reviewed` project state and curated for publication as JxAgent. Generated audit payloads, machine-local reports, datasets, credentials, model weights, and checkpoints were excluded.

Validation performed while packaging:

- pytest collection: 462 tests discovered successfully
- second-stage hardening suite: 17 passed
- training infrastructure suite: 37 passed
- core schema/coordinates/decontamination/dedup/sampling/splits/evaluation subset: 90 passed
- reproducibility/prebuild/quality-hardening subset: 91 passed
- publication safety scan: passed

Total explicitly executed targeted tests: **235 passed**.

The GitHub Actions workflow runs the full test suite on push and pull requests. A packaging-time timeout of the full suite is not treated as a full-suite pass, so this document does not claim one.
