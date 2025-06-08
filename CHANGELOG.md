# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-06-08

### Added

- First beta release of pyvers 🎉
- Dynamic dispatch based on module versions and backends
- Support for version-specific implementations using `@implement_for` decorator
- Backend switching with `set_backend` context manager
- Backend registration system with `register_backend`
- Dynamic backend access with `get_backend`
- Example implementations:
  - NumPy 1.x vs 2.0 type changes (bool8/bool_)
  - PyTorch 2.0+ optimizations with torch.compile
  - SciPy/CuPy sparse matrix operations (CPU vs GPU)
  - Gym/Gymnasium API compatibility

### Features

- Automatic version detection and compatibility checking
- Clean API for managing multiple backend implementations
- Support for Python 3.9-3.13
- Comprehensive test suite with high coverage
- Well-documented examples in the examples/ directory

### Dependencies

- Core package has minimal dependencies
- Optional test dependencies for development
- Example dependencies:
  - NumPy
  - JAX
  - PyTorch
  - SciPy/CuPy
  - Gym/Gymnasium

### Note

This is a beta release. While the core functionality is stable, the API may undergo minor changes based on user feedback before reaching 1.0.0. 