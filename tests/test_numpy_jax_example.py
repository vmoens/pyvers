"""Test the NumPy/JAX example from the README."""

import numpy as np
import pytest

from pyvers import get_backend, implement_for, register_backend, set_backend


# Check if NumPy version is compatible with JAX's requirements
# JAX >= 0.4.30 requires NumPy >= 2.0 for some internal operations
def _check_jax_numpy_compatibility():
    """Check if JAX and NumPy versions are compatible."""
    try:
        import jax
        import jax.numpy as jnp

        # Try a jitted operation that would fail with incompatible versions
        # The issue is when JAX's jit interacts with NumPy < 2.0
        @jax.jit
        def _test_fn(x):
            return jnp.greater(x, 0)

        arr = jnp.asarray([1, 2, 3])
        result = _test_fn(arr)
        # This triggers __array__ which uses copy= kwarg requiring NumPy 2.0+
        np.asarray(result)
        return True
    except (ImportError, TypeError):
        return False


@pytest.mark.skipif(
    not _check_jax_numpy_compatibility(),
    reason="JAX and NumPy versions are incompatible",
)
def test_numpy_jax_example():
    """Test the NumPy/JAX example from the README."""
    import jax.numpy as jnp

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

    # Test data
    test_arr = [-1, 2, -3, 4]
    expected = [False, True, False, True]

    # Test NumPy implementation - use numpy backend explicitly
    # since JAX was registered last and would be the default
    with set_backend("numpy", "numpy"):
        numpy_result = create_mask(test_arr)
        np.testing.assert_array_equal(numpy_result, expected)

    # Test JAX implementation
    with set_backend("numpy", "jax.numpy"):
        jax_result = create_mask(test_arr)
        np.testing.assert_array_equal(jax_result, expected)


if __name__ == "__main__":
    test_numpy_jax_example()
