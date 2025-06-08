"""Tests for backend management system."""

import pytest

from pyvers.backends import BackendManager, register_backend


def test_duplicate_backend_registration():
    """Test that registering the same backend in different groups raises an error."""
    # clean up the registered backends for other tests
    BackendManager._registered_backends.clear()

    # First registration should succeed
    register_backend("group1", {
        "numpy": "numpy",
        "backend1": "some.module1",
    })

    # Second registration with same backend name should fail
    with pytest.raises(ValueError, match="Backend 'numpy' is already registered in group 'group1'"):
        register_backend("group2", {
            "numpy": "different.numpy",  # Same backend name as in group1
            "backend2": "some.module2",
        })


    # Clean up the registered backends for other tests
    BackendManager._registered_backends.clear()


def test_valid_backend_registration():
    """Test that registering different backends in different groups works."""
    # clean up the registered backends for other tests
    BackendManager._registered_backends.clear()

    register_backend("group1", {
        "backend1": "some.module1",
    })

    # This should work as it uses different backend names
    register_backend("group2", {
        "backend2": "some.module2",
    })

    assert BackendManager._registered_backends["backend1"] == "group1"
    assert BackendManager._registered_backends["backend2"] == "group2"

    # Clean up the registered backends for other tests
    BackendManager._registered_backends.clear()
