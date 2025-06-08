# WARNING: This example requires two separate environments:
#  - One with numpy < 2.0 (e.g. numpy 1.24.0)
#  - One with numpy >= 2.0 (e.g. numpy 2.0.0)
# The example demonstrates how to handle the bool8 vs bool_ type change
# between numpy 1 and 2 while maintaining consistent behavior.

from pyvers import get_backend, implement_for, register_backend

# Register numpy backend versions
register_backend(group="numpy", backends={"numpy": "numpy"})

# Function to create a boolean mask array
# In numpy 1.x, we can use bool8 which was the explicit type for boolean arrays
@implement_for("numpy", from_version=None, to_version="2.0.0")
def create_boolean_mask(arr):
    print(f"Using numpy {get_backend('numpy').__version__} (pre-2.0)")
    np = get_backend("numpy")
    # In numpy 1.x, bool8 was commonly used for boolean arrays
    return np.array([x > 0 for x in arr], dtype=np.bool8)

# In numpy 2.x, bool8 was removed in favor of just bool_
@implement_for("numpy", from_version="2.0.0")
def create_boolean_mask(arr):  # noqa: F811
    print(f"Using numpy {get_backend('numpy').__version__} (2.0+)")
    np = get_backend("numpy")
    # In numpy 2.x, we use bool_ as bool8 was removed
    return np.array([x > 0 for x in arr], dtype=np.bool_)

if __name__ == "__main__":
    import numpy as np
    test_array = [-1, 2, -3, 4, -5]

    # The function will automatically use the appropriate implementation
    # based on the numpy version, but the results will be the same
    mask = create_boolean_mask(test_array)
    print(f"Boolean mask: {mask}")
    print(f"Mask dtype: {mask.dtype}")

    # Verify the mask works the same in both versions
    filtered = np.array(test_array)[mask]
    print(f"Filtered positive numbers: {filtered}")
    assert all(x > 0 for x in filtered)

    print("\n" + "="*80)
    if float(np.__version__.split('.')[0]) >= 2:
        print("You're running this with NumPy 2+.")
        print("Try creating a new environment with NumPy < 2.0 (e.g. 1.24.0) and run this again!")
        print("Note: With NumPy >= 1.24.0, you'll see a deprecation warning about bool8 - that's expected!")
    else:
        print("You're running this with NumPy 1.x.")
        print("Try creating a new environment with NumPy >= 2.0 and run this again!")
        if float(np.__version__.split('.')[1]) >= 24:
            print("Note: The deprecation warning about bool8 above is expected in NumPy >= 1.24.0!")

    # Note for users:
    # Try running this script in different environments to see how pyvers handles the version differences:
    #  - With numpy < 2.0: Will use bool8 (Note: if numpy >= 1.24.0, you'll see a deprecation warning about bool8)
    #  - With numpy >= 2.0: Will use bool_ (the new standard way)
    # The results will be functionally identical in both cases!
