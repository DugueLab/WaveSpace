# Changelog

All notable changes to WaveSpace are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.9] - 2026-08-19

### Added
- tox configuration to test Python 3.9–3.14.
- Sphinx-Gallery documentation generated from the example scripts.
- GitHub Actions workflow for documentation builds and updated checks for tests, dependencies, and releases.
- Declared `joblib` and `vtk` as required runtime dependencies.

### Changed
- Modernized project packaging and development environment configuration with `pyproject.toml` and `uv`.
- Updated examples, API documentation, and contributor guidance.
- Enforced Ruff checks for import ordering, common bugs, Python modernization, simplifications, comprehensions, and Ruff-specific rules.
- Standardized public documentation on NumPy-style docstrings.

### Removed
- Removed `ClusterGradient`, `MEMD_Matlab_translation`, and `generateFromLog` modules, as they were still in development.

## [1.1.8] - 2026-08-12

### Added
- Initial release published on PyPI.
- `WaveData` container class for multi-channel neural recordings.
- Simulation, preprocessing, decomposition, spatial arrangement, wave analysis, plotting helpers, statistics, and utils modules.
- Sphinx documentation hosted on Read the Docs.
- GitHub Actions workflows for tests (Python 3.9-3.14) and PyPI release on tag.
