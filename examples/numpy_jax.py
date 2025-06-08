# Requires numpy and jax to be installed:
#  !pip install numpy "jax[cpu]"

from pyvers import get_backend, implement_for, register_backend, set_backend

# First, we register the backends: the backends must be a dictionary of the form {backend_name: module_name}
register_backend(group="numpy", backends={"numpy": "numpy", "jax.numpy": "jax.numpy"})

# alternative:
# import numpy
# import jax.numpy
# register_backend(group="numpy", backends={"numpy": numpy, "jax.numpy": jax.numpy})

# Then, we implement the function for the jax.numpy backend
@implement_for("jax.numpy", from_version="0.4.0")
def matrix_operation(x, y, np):
    # We only need jax directly for grad
    import jax
    result = np.matmul(x, y)
    gradient = jax.grad(lambda x, y: np.sum(np.matmul(x, y)))(x, y)
    return result, gradient

# Then, we implement the function for the numpy backend
@implement_for("numpy")
def matrix_operation(x, y, np):  # noqa: F811
    return np.matmul(x, y), None


if __name__ == "__main__":
    # Create some test matrices
    import numpy as onp  # original numpy for data creation
    x = onp.array([[1., 2.], [3., 4.]])
    y = onp.array([[5., 6.], [7., 8.]])

    print(f"Using numpy backend: {get_backend('numpy')}")
    # Check that we use the numpy backend
    with set_backend("numpy", "numpy"):
        print(f"Using numpy backend: {get_backend('numpy')} within the numpy context manager")
        np = get_backend("numpy")
        result, gradient = matrix_operation(x, y, np)
        print("NumPy result:", result)
        print("NumPy gradient (should be None):", gradient)

    print(f"\nUsing numpy backend: {get_backend('numpy')}")
    # Check that we use the jax.numpy backend
    with set_backend("numpy", "jax.numpy"):
        print(f"Using numpy backend: {get_backend('numpy')} within the jax.numpy context manager")
        np = get_backend("numpy")
        result, gradient = matrix_operation(x, y, np)
        print("JAX result:", result)
        print("JAX gradient:", gradient)

    print(f"\nUsing numpy backend: {get_backend('numpy')}")
    # Check that we use the numpy backend again
    with set_backend("numpy", "numpy"):
        print(f"Using numpy backend: {get_backend('numpy')} within the numpy context manager")
        np = get_backend("numpy")
        result, gradient = matrix_operation(x, y, np)
        print("NumPy result:", result)
        print("NumPy gradient (should be None):", gradient)
