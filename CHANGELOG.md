# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-01-23

### Fixed

- Fix `module_set()` to return raw function instead of wrapper
  - After `module_set()` is called, the module attribute is now the raw function instead of a `_RegisterableFunction` wrapper
  - This fixes identity comparison with `is` which was failing in downstream tests
  - The `.register()` method is only needed during initial decoration setup, not after the function has been called

## [0.2.1] - 2026-01-22

### Fixed

- Fix descriptor protocol for `_RegisterableFunction` to properly bind `self` for instance methods
  - Added `__get__` method so `@implement_for` decorated methods work correctly as instance methods
  - Without this fix, instance methods would fail with "missing 1 required positional argument"

## [0.2.0] - 2026-01-22

### Added

- New `.register()` API for `@implement_for` decorator, following the `functools.singledispatch` pattern
  - Allows registering version-specific implementations without linter warnings
  - Use `_` as the function name for registered implementations (recognized by linters)
  - Example:
    ```python
    @implement_for("numpy")
    def process_array(arr):
        raise NotImplementedError("No matching version")

    @process_array.register(from_version="2.0.0")
    def _(arr):
        # numpy >= 2.0 implementation
        return arr * 3
    ```

### Changed

- Removed `packaging` dependency - pyvers now has zero external dependencies
  - Replaced `packaging.version.parse` with a lightweight internal `_parse_version` function
  - Resolves version conflicts with projects that pin older versions of `packaging`

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