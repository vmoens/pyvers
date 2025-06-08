# WARNING: This example requires two separate environments:
#  - One with scipy installed (CPU backend)
#  - One with cupy installed (GPU backend)
# The example demonstrates how to use the same sparse matrix operations
# on either CPU (scipy) or GPU (cupy) while maintaining the same API.

import numpy

from pyvers import get_backend, implement_for, register_backend, set_backend

# Register both scipy and cupy backends
# Note: Since both the key and value must match the module name, this could potentially
# be simplified in the future to just accept a list of strings: ["scipy.sparse", "cupyx.scipy.sparse"]
register_backend(group="sparse", backends={
    "scipy.sparse": "scipy.sparse",        # CPU backend
    "cupyx.scipy.sparse": "cupyx.scipy.sparse"  # GPU backend - does NOT require cupy to be installed!
})
# alternative (requires scipy and cupy to be installed):
# import scipy.sparse
# import cupyx.scipy.sparse
# register_backend(group="sparse", backends={"scipy.sparse": scipy.sparse, "cupyx.scipy.sparse": cupyx.scipy.sparse})

# This example demonstrates how to handle different sparse matrix implementations
# between SciPy (CPU) and CuPy (GPU) while maintaining consistent results.
# It shows how to work around implementation differences (like maximum/minimum
# operations which are handled differently in the two libraries).

# Function to find maximum values in sparse matrix blocks
# The implementations differ because:
# - SciPy sparse matrices support direct maximum operations
# - CuPy sparse matrices need to be converted to dense for this operation
@implement_for("scipy.sparse")
def block_max_sparse(size=1000, block_size=100, density=0.01):
    print("Using SciPy sparse (CPU backend)")
    sparse = get_backend("sparse")
    np = numpy

    # Create a random sparse matrix
    A = sparse.random(size, size, density=density, format='csr', dtype=np.float32, random_state=42)  # noqa: N806

    # In SciPy, we can directly get max values from sparse matrix blocks
    n_blocks = size // block_size
    block_maxes = np.zeros((n_blocks, n_blocks), dtype=np.float32)

    for i in range(n_blocks):
        for j in range(n_blocks):
            start_i = i * block_size
            start_j = j * block_size
            block = A[start_i:start_i + block_size, start_j:start_j + block_size]
            # SciPy sparse matrices support direct max operation
            block_maxes[i, j] = block.max()

    return {
        'nnz': A.nnz,
        'block_maxes': block_maxes,
        'global_max': block_maxes.max()
    }

@implement_for("cupyx.scipy.sparse")
def block_max_sparse(size=1000, block_size=100, density=0.01):  # noqa: F811
    print("Using CuPy sparse (GPU backend)")
    sparse = get_backend("sparse")
    cp = get_backend("cupy")  # For array operations

    # Create a random sparse matrix
    A = sparse.random(size, size, density=density, format='csr', dtype=cp.float32, random_state=42)  # noqa: N806

    # In CuPy, we need to handle blocks differently since sparse matrices
    # don't support direct max operation
    n_blocks = size // block_size
    block_maxes = cp.zeros((n_blocks, n_blocks), dtype=cp.float32)

    for i in range(n_blocks):
        for j in range(n_blocks):
            start_i = i * block_size
            start_j = j * block_size
            block = A[start_i:start_i + block_size, start_j:start_j + block_size]
            # Need to convert to dense for max operation in CuPy
            block_maxes[i, j] = block.todense().max()

    # Convert results back to CPU for comparison
    return {
        'nnz': A.nnz,
        'block_maxes': cp.asnumpy(block_maxes),
        'global_max': float(block_maxes.max())
    }

if __name__ == "__main__":
    # Parameters for our test
    SIZE = 500
    BLOCK_SIZE = 100
    DENSITY = 0.05

    print("\nTesting with SciPy backend...")
    with set_backend("sparse", "scipy.sparse"):
        scipy_results = block_max_sparse(SIZE, BLOCK_SIZE, DENSITY)

    print("\nTesting with CuPy backend...")
    try:
        with set_backend("sparse", "cupyx.scipy.sparse"):
            cupy_results = block_max_sparse(SIZE, BLOCK_SIZE, DENSITY)

        # Compare results
        print("\nComparing results:")
        print(f"SciPy non-zero elements: {scipy_results['nnz']}")
        print(f"CuPy non-zero elements: {cupy_results['nnz']}")
        print(f"SciPy global maximum: {scipy_results['global_max']:.4f}")
        print(f"CuPy global maximum: {cupy_results['global_max']:.4f}")

        # Check if block maxima are close (they might not be identical due to GPU/CPU differences)
        max_diff = numpy.abs(scipy_results['block_maxes'] - cupy_results['block_maxes']).max()
        print(f"\nMaximum difference between block maxima: {max_diff:.6f}")
        print("(Small differences are expected due to random initialization and floating-point arithmetic)")

    except ImportError:
        print("\nCuPy is not installed. To see the GPU version:")
        print("1. Create a new environment")
        print("2. Install CuPy: pip install cupy-cuda11x (replace with your CUDA version)")
        print("3. Run this script again")
        print("\nNote: The script will still work with just SciPy, but you won't see the comparison")

    # Check which backend we're using by attempting to import cupy
    has_cupy = False
    try:
        import cupy  # noqa: F401
        has_cupy = True
    except ImportError:
        pass

    if has_cupy:
        print("You're running this with CuPy (GPU backend).")
        print("Try running this in an environment with only SciPy to see the CPU version!")
        print("\nNote: The operations are identical, but the GPU version might be faster")
        print("for larger matrices, especially if you increase the 'size' parameter.")
    else:
        print("You're running this with SciPy (CPU backend).")
        print("Try installing CuPy in a new environment to see the GPU version!")
        print("\nNote: With CuPy, these operations would run on the GPU,")
        print("potentially much faster for larger matrices.")
