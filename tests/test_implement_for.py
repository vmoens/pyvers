from __future__ import annotations

import argparse
import sys
from copy import copy
from importlib import import_module
from unittest import mock

import _utils_internal
import pytest

from pyvers import implement_for
from pyvers.backends import set_gym_backend
from pyvers.implement_for import _RegisterableFunction


def _unwrap_fn(fn):
    """Unwrap a _RegisterableFunction to get the underlying callable."""
    if isinstance(fn, _RegisterableFunction):
        return fn._fn
    return fn


def uncallable(f):
    class UncallableObject:
        def __init__(self, other):
            for k, v in other.__dict__.items():
                if k not in ("__call__", "__dict__", "__weakref__"):
                    setattr(self, k, v)

    g = UncallableObject(f)
    return g


class implement_for_test_functions:  # noqa: N801
    """
    Groups functions that are used in tests for `implement_for` decorator.
    """

    @staticmethod
    @implement_for(lambda: import_module("_utils_internal"), "0.3")
    def select_correct_version():
        """To test from+ range and that this function is not selected as the implementation."""
        return "0.3+V1"

    @staticmethod
    @implement_for("_utils_internal", "0.3")
    def select_correct_version():  # noqa: F811
        """To test that this function is selected as the implementation (last implementation)."""
        return "0.3+"

    @staticmethod
    @implement_for(lambda: import_module("_utils_internal"), "0.2", "0.3")
    def select_correct_version():  # noqa: F811
        """To test that right bound is not included."""
        return "0.2-0.3"

    @staticmethod
    @implement_for("_utils_internal", "0.1", "0.2")
    def select_correct_version():  # noqa: F811
        """To test that function with missing from-to range is ignored."""
        return "0.1-0.2"

    @staticmethod
    @implement_for("missing_module")
    def missing_module():
        """To test that calling decorated function with missing module raises an exception."""
        return "missing"

    @staticmethod
    @implement_for("_utils_internal", None, "0.3")
    def missing_version():
        return "0-0.3"

    @staticmethod
    @implement_for("_utils_internal", "0.4")
    def missing_version():  # noqa: F811
        return "0.4+"


# Test the new register API - no noqa needed!
# This is a module-level function demonstrating the singledispatch-style API
@implement_for("_utils_internal")
def register_api_test(x):
    """Base function that should not be called if a matching version exists."""
    raise NotImplementedError("No matching version")


@register_api_test.register(from_version="0.3")
def _(x):
    return f"register_0.3+:{x}"


@register_api_test.register(from_version="0.2", to_version="0.3")
def _(x):
    return f"register_0.2-0.3:{x}"


# Separate function for testing .register() method availability
# This function should NOT be called by other tests to preserve the wrapper
@implement_for("_utils_internal")
def register_method_test(x):
    """Function for testing .register() method availability."""
    raise NotImplementedError("No matching version")


@register_method_test.register(from_version="0.3")
def _(x):
    return f"test:{x}"


def test_implement_for():
    assert implement_for_test_functions.select_correct_version() == "0.3+"


def test_implement_for_missing_module():
    msg = r"Supported version of 'test_implement_for.implement_for_test_functions.missing_module' has not been found."
    with pytest.raises(ModuleNotFoundError, match=msg):
        implement_for_test_functions.missing_module()


def test_implement_for_missing_version():
    msg = r"Supported version of 'test_implement_for.implement_for_test_functions.missing_version' has not been found."
    with pytest.raises(ModuleNotFoundError, match=msg):
        implement_for_test_functions.missing_version()


def test_register_api():
    """Test the singledispatch-style .register() API."""
    # _utils_internal has version 0.3, so the "0.3+" implementation should be used
    result = register_api_test("hello")
    assert result == "register_0.3+:hello"


def test_register_api_has_register_method():
    """Test that decorated functions have a .register() method.

    Note: The .register() method is only available before the function is first
    called. After the first call, module_set() replaces the _RegisterableFunction
    wrapper with the raw function for identity comparison compatibility.
    """
    # Use register_method_test which is not called by other tests
    assert hasattr(register_method_test, "register")
    assert callable(register_method_test.register)


def test_register_api_preserves_name():
    """Test that the register API preserves the original function name."""
    assert register_api_test.__name__ == "register_api_test"


class TestInstanceMethodBinding:
    """Test that @implement_for works correctly on instance methods."""

    @implement_for("_utils_internal", "0.3")
    def instance_method(self, x):
        return f"instance:{x}:self={self.__class__.__name__}"

    def test_instance_method_binding(self):
        """Test that self is properly bound when calling instance methods."""
        result = self.instance_method("hello")
        assert result == "instance:hello:self=TestInstanceMethodBinding"

    def test_instance_method_from_instance(self):
        """Test calling instance method from an instance variable."""
        obj = TestInstanceMethodBinding()
        result = obj.instance_method("world")
        assert result == "instance:world:self=TestInstanceMethodBinding"


def test_implement_for_reset():
    assert implement_for_test_functions.select_correct_version() == "0.3+"
    _impl = copy(implement_for._implementations)
    name = implement_for.get_func_name(
        implement_for_test_functions.select_correct_version
    )
    for setter in implement_for._setters:
        if implement_for.get_func_name(setter.fn) == name and setter.fn() != "0.3+":
            setter.module_set()
    assert implement_for_test_functions.select_correct_version() != "0.3+"
    implement_for.reset(_impl)
    assert implement_for_test_functions.select_correct_version() == "0.3+"


def test_module_set_returns_raw_function():
    """Test that after module_set(), the module attribute is the raw function.

    This is a regression test for an issue where module_set() would wrap the
    function in _RegisterableFunction, breaking identity comparison with `is`.
    See: https://github.com/pytorch/rl test_set_gym_environments failures.
    """
    # Get an implementation that matches the current version
    matching_setter = None
    for setter in implement_for._setters:
        if setter.fn.__name__ == "_set_gym_environments":
            # Check if this implementation matches (we'll use any gymnasium impl)
            if setter.module_name == "gymnasium":
                matching_setter = setter
                break

    if matching_setter is None:
        pytest.skip("No matching _set_gym_environments implementation found")

    # Call module_set() to set the function on the module
    matching_setter.do_set = True
    matching_setter.module_set()

    # After module_set(), the module attribute should be the raw function,
    # NOT a _RegisterableFunction wrapper. This allows identity comparison with `is`.
    module_attr = getattr(_utils_internal, "_set_gym_environments", None)
    assert module_attr is not None, "Function not set on module"
    assert not isinstance(
        module_attr, _RegisterableFunction
    ), f"Expected raw function, got {type(module_attr)}"
    assert module_attr is matching_setter.fn, "Function identity mismatch"


@pytest.mark.parametrize(
    "version, from_version, to_version, expected_check",
    [
        ("0.21.0", "0.21.0", None, True),
        ("0.21.0", None, "0.21.0", False),
        ("0.9.0", "0.11.0", "0.21.0", False),
        ("0.9.0", "0.1.0", "0.21.0", True),
        ("0.19.99", "0.19.9", "0.21.0", True),
        ("0.19.99", None, "0.19.0", False),
        ("0.99.0", "0.21.0", None, True),
        ("0.99.0", None, "0.21.0", False),
    ],
)
def test_implement_for_check_versions(
    version, from_version, to_version, expected_check
):
    assert (
        implement_for.check_version(version, from_version, to_version) == expected_check
    )


@pytest.mark.parametrize(
    "gymnasium_version, expected_from_version_gymnasium, expected_to_version_gymnasium",
    [
        ("0.27.0", None, "1.0.0"),
        ("0.27.2", None, "1.0.0"),
        # ("1.0.1", "1.0.0", None),
    ],
)
@pytest.mark.parametrize(
    "gym_version, expected_from_version_gym, expected_to_version_gym",
    [
        ("0.21.0", "0.21.0", None),
        ("0.22.0", "0.21.0", None),
        ("0.99.0", "0.21.0", None),
        ("0.9.0", None, "0.21.0"),
        ("0.20.0", None, "0.21.0"),
        ("0.19.99", None, "0.21.0"),
    ],
)
def test_set_gym_environments(
    gym_version,
    expected_from_version_gym,
    expected_to_version_gym,
    gymnasium_version,
    expected_from_version_gymnasium,
    expected_to_version_gymnasium,
):
    # Save original modules to restore after the test
    original_gym = sys.modules.get("gym")
    original_gymnasium = sys.modules.get("gymnasium")

    try:
        # mock gym and gymnasium imports
        mock_gym = uncallable(mock.MagicMock())
        mock_gym.__version__ = gym_version
        mock_gym.__name__ = "gym"
        sys.modules["gym"] = mock_gym

        mock_gymnasium = uncallable(mock.MagicMock())
        mock_gymnasium.__version__ = gymnasium_version
        mock_gymnasium.__name__ = "gymnasium"
        sys.modules["gymnasium"] = mock_gymnasium

        import gym
        import gymnasium

        # look for the right function that should be called according to gym versions
        # (and same for gymnasium)
        expected_fn_gymnasium = None
        expected_fn_gym = None
        for impfor in implement_for._setters:
            if impfor.fn.__name__ == "_set_gym_environments":
                if (impfor.module_name, impfor.from_version, impfor.to_version) == (
                    "gym",
                    expected_from_version_gym,
                    expected_to_version_gym,
                ):
                    expected_fn_gym = impfor.fn
                elif (impfor.module_name, impfor.from_version, impfor.to_version) == (
                    "gymnasium",
                    expected_from_version_gymnasium,
                    expected_to_version_gymnasium,
                ):
                    expected_fn_gymnasium = impfor.fn
                if expected_fn_gym is not None and expected_fn_gymnasium is not None:
                    break

        with set_gym_backend(gymnasium):
            # The module attribute may be wrapped in _RegisterableFunction
            actual_fn = _unwrap_fn(_utils_internal._set_gym_environments)
            assert actual_fn is expected_fn_gymnasium, expected_fn_gym

        with set_gym_backend(gym):
            actual_fn = _unwrap_fn(_utils_internal._set_gym_environments)
            assert actual_fn is expected_fn_gym, expected_fn_gymnasium

        with set_gym_backend(gymnasium):
            actual_fn = _unwrap_fn(_utils_internal._set_gym_environments)
            assert actual_fn is expected_fn_gymnasium, expected_fn_gym
    finally:
        # Restore original modules to avoid polluting other tests
        if original_gym is not None:
            sys.modules["gym"] = original_gym
        else:
            sys.modules.pop("gym", None)
        if original_gymnasium is not None:
            sys.modules["gymnasium"] = original_gymnasium
        else:
            sys.modules.pop("gymnasium", None)
        # Clear implement_for's module cache to avoid stale cached mock modules
        implement_for._cache_modules.pop("gym", None)
        implement_for._cache_modules.pop("gymnasium", None)


if __name__ == "__main__":
    import argparse

    import pytest

    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
