# 🐦 pyvers

A Python library for dynamic dispatch based on module versions and backends.

## What can you do with pyvers?

- 🔄 Handle breaking changes between different versions of a library without cluttering your code 
- 🔀 Switch seamlessly between different backend implementations (e.g., CPU vs GPU)
- ✨ Support multiple versions of a dependency in the same codebase without complex if/else logic 
- 🚀 Write version-specific optimizations while maintaining backward compatibility
- 🧹 Keep your code clean and maintainable while supporting multiple environments

## Usage

pyvers lets you write version-specific implementations that are automatically selected based on the installed package version or backend. Here's a simple example:

```python
from pyvers import implement_for, register_backend, get_backend, set_backend

# Register numpy backend - you could register more than one backend!
register_backend(group="numpy", backends={"numpy": "numpy"})

# Function for NumPy < 2.0 (using bool8)
@implement_for("numpy", from_version=None, to_version="2.0.0")
def create_mask(arr):
    np = get_backend("numpy")
    return np.array([x > 0 for x in arr], dtype=np.bool8)

# Function for NumPy >= 2.0 (using bool_)
@implement_for("numpy", from_version="2.0.0")
def create_mask(arr):  # noqa: F811
    np = get_backend("numpy")
    return np.array([x > 0 for x in arr], dtype=np.bool_)

# Implement a jax version of this
register_backend(group="numpy", backends={"jax.numpy": "jax.numpy"})
register_backend(group="jax", backends={"jax": "jax"})  # For version checking

@implement_for("jax")  # Check jax version instead of jax.numpy
def create_mask(arr):  # noqa: F811
    import jax
    import jax.numpy as jnp  # Import directly for clarity
    
    # Use JAX's JIT compilation and vectorization
    @jax.jit
    def _create_mask(x):
        return jnp.greater(x, 0).astype(jnp.bool_)
    
    return _create_mask(jnp.asarray(arr))

# The correct implementation is automatically chosen based on your NumPy version
result = create_mask([-1, 2, -3, 4])
print("NumPy result:", result)

with set_backend("numpy", "jax.numpy"):
    result = create_mask([-1, 2, -3, 4])
    print("JAX result:", result)

```

Check out the [examples](examples/) folder for more advanced use cases:
- Switching between NumPy and JAX.numpy backends
- Handling CPU (SciPy) vs GPU (CuPy) implementations
- Managing breaking changes in PyTorch 2.0
- Supporting both gym and gymnasium APIs

## Installation

```bash
pip install pyvers
```

## Features

### Version-based dispatch

Automatically select the right implementation based on package versions:
```python
@implement_for("torch", from_version="2.0.0")
def optimize_model(model):
    return torch.compile(model)  # Only available in PyTorch 2.0+

@implement_for("torch", from_version=None, to_version="2.0.0")
def optimize_model(model):  # noqa: F811
    return model  # Fallback for older versions
```

### Backend switching

Easily switch between different implementations:
```python
# Register both backends
register_backend(group="numpy", backends={
    "numpy": "numpy",
    "jax.numpy": "jax.numpy"
})

# Use context manager to switch backends
with set_backend("numpy", "jax.numpy"):
    result = your_function()  # Uses JAX
with set_backend("numpy", "numpy"):
    result = your_function()  # Uses NumPy
```

### Dynamic imports
Backends are imported only when needed, so you can have optional dependencies:
```python
register_backend(group="sparse", backends={
    "scipy.sparse": "scipy.sparse",        # CPU backend
    "cupyx.scipy.sparse": "cupyx.scipy.sparse"  # GPU backend - does NOT require cupy to be installed!
})
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Development

### Setup

1. Clone the repository
2. Install Poetry (package manager)
3. Install dependencies:
   ```bash
   poetry install
   ```

### Running Tests

```bash
poetry run pytest
```

This will run the test suite with coverage reporting.

### Code Quality

We use [Ruff](https://github.com/astral-sh/ruff) for linting and code formatting. Ruff combines multiple Python linters into a single fast, unified tool.

To check your code:
```bash
poetry run ruff check .
```

To automatically fix issues:
```bash
poetry run ruff check --fix .
```

Ruff is configured to:
- Follow PEP 8 style guide
- Sort imports automatically
- Check for common bugs and code complexity
- Target Python 3.12+

See `pyproject.toml` for the complete linting configuration.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

## Citation

pyvers was developped as part of [TorchRL](https://github.com/pytorch/rl).
