# Backend management system for dynamic dispatch.

from __future__ import annotations

import collections
import contextlib
import importlib
from collections.abc import Callable
from copy import copy
from functools import wraps
from typing import Any

from .implement_for import implement_for

# Global registry for default backends
_DEFAULT_BACKENDS: dict[str, Any] = {}
_IMPORT_ERRORS: dict[str, Exception] = {}

class BackendManager:
    """Context manager and decorator for setting backends.

    Args:
        group (str): The backend group (e.g., "gym", "numpy", etc.)
        backend (str, module, or callable): The backend to use. Can be:
            - A string (module name to import)
            - A module object
            - A callable that returns a module

    Version checking:
        When checking versions for submodules (e.g. jax.numpy), the version of the base package
        (e.g. jax) will be used. This is because submodules typically don't have their own version
        numbers and follow the main package's versioning.

        For example:
            >>> register_backend("numpy", {"jax.numpy": "jax.numpy"})
            >>> # This will check jax.__version__, not jax.numpy.__version__
            >>> @implement_for("jax.numpy", from_version="0.4.0")
            >>> def my_function(): ...
    """
    # Store all registered backends to check for duplicates
    _registered_backends: dict[str, str] = {}  # maps backend_name -> group_name

    def __init__(self, group: str, backend: str | Any | Callable[[], Any]):
        self.group = group
        self._backend = backend
        self._setters_saved = None

    def _get_base_package(self, module: Any) -> Any:
        """Get the base package of a module for version checking.

        For example:

            - For jax.numpy, returns jax
            - For numpy, returns numpy
        """
        # Get the root package name (e.g. 'jax' from 'jax.numpy')
        root_name = module.__name__.split('.')[0]
        if root_name != module.__name__:
            # If this is a submodule, import and return the root package
            return importlib.import_module(root_name)
        return module

    def _call(self) -> None:
        """Sets the backend as default for its group."""
        global _DEFAULT_BACKENDS
        _DEFAULT_BACKENDS[self.group] = self.backend

        found_setters = collections.defaultdict(lambda: False)
        for setter in copy(implement_for._setters):
            # Get the base package for version checking
            base_package = self._get_base_package(self.backend)

            check_module = (
                callable(setter.module_name)
                and setter.module_name.__name__ == self.backend.__name__
            ) or setter.module_name == self.backend.__name__

            check_version = setter.check_version(
                base_package.__version__, setter.from_version, setter.to_version
            )

            if check_module and check_version:
                setter.module_set()
                found_setter = True
            elif check_module:
                found_setter = False
            else:
                found_setter = None

            if found_setter is not None:
                found_setters[setter.func_name] = (
                    found_setters[setter.func_name] or found_setter
                )

        # Verify that all required setters were found
        for func_name, found_setter in found_setters.items():
            if not found_setter:
                base_package = self._get_base_package(self.backend)
                raise ImportError(
                    f"Could not set backend {self.backend.__name__} "
                    f"(version={base_package.__version__}) "
                    f"for function {func_name} in group {self.group}. "
                    f"Check version compatibility!"
                )

    def set(self) -> None:
        """Irreversibly sets the backend for this group."""
        prev_backend = _DEFAULT_BACKENDS.get(self.group)
        _DEFAULT_BACKENDS[self.group] = self.backend
        self._call()
        return prev_backend

    def reset(self, prev_backend: Any) -> None:
        """Resets the backend for this group."""
        _DEFAULT_BACKENDS[self.group] = prev_backend
        self._call()

    def __enter__(self) -> None:
        self._setters_saved = copy(implement_for._implementations)
        self._call()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        implement_for.reset(setters_dict=self._setters_saved)
        delattr(self, "_setters_saved")

    def __call__(self, func: Callable) -> Callable:
        """Allows using this as a decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

    @property
    def backend(self) -> Any:
        """Gets the actual backend module."""
        if isinstance(self._backend, str):
            return importlib.import_module(self._backend)
        elif callable(self._backend):
            return self._backend()
        return self._backend

    @backend.setter
    def backend(self, value: str | Any | Callable[[], Any]) -> None:
        self._backend = value


def register_backend(group: str, backends: dict[str, str | Any | Callable[[], Any]]) -> None:
    """Register available backends for a group.

    Args:
        group: The backend group (e.g., "gym", "numpy")
        backends: Dictionary mapping backend names to their implementations
            Each implementation can be:
            - A string (module name to import)
            - A module object
            - A callable that returns a module

    Raises:
        ValueError: If a backend is already registered in another group.

    Examples:
        >>> register_backend("gym", {
        ...     "gym": "gym",
        ...     "gymnasium": "gymnasium",
        ... })
        >>> register_backend("numpy", {
        ...     "numpy": "numpy",
        ...     "jax.numpy": lambda: import_module("jax.numpy"),  # Will check jax version
        ... })
    """
    if not isinstance(backends, dict):
        raise TypeError("backends must be a dictionary")

    # Check for duplicate backends
    for backend_name in backends:
        if backend_name in BackendManager._registered_backends:
            existing_group = BackendManager._registered_backends[backend_name]
            if existing_group != group:
                raise ValueError(
                    f"Backend '{backend_name}' is already registered in group '{existing_group}'. "
                    f"Cannot register it again in group '{group}'."
                )
        BackendManager._registered_backends[backend_name] = group

    # Store the backends configuration for the group
    setattr(BackendManager, f"_{group}_backends", backends)


@contextlib.contextmanager
def set_backend(group: str, backend: str | Any | Callable[[], Any]) -> Any:
    """Set the default backend for a group.

    Args:
        group: The backend group (e.g., "gym", "numpy")
        backend: The backend to use. Can be:
            - A string (module name to import)
            - A module object
            - A callable that returns a module

    Examples:
        >>> set_backend("gym", "gymnasium")  # Use gymnasium
        >>> set_backend("numpy", "numpy")    # Use numpy
        >>> set_backend("numpy", lambda: import_module("jax.numpy"))  # Use JAX
    """
    manager = BackendManager(group, backend)
    prev_backend = manager.set()
    yield
    manager.reset(prev_backend)


def get_backend(group: str, submodule: str | None = None) -> Any:
    """Get the current backend for a group.

    Args:
        group: The backend group (e.g., "gym", "numpy")
        submodule: Optional submodule to import from the backend

    Returns:
        The backend module or submodule

    Examples:
        >>> np = get_backend("numpy")  # Get the current numpy backend
        >>> gym = get_backend("gym")   # Get the current gym backend
        >>> wrappers = get_backend("gym", "wrappers")  # Get gym.wrappers
    """
    global _DEFAULT_BACKENDS, _IMPORT_ERRORS

    if group not in _DEFAULT_BACKENDS:
        # Try to import registered backends in order until one succeeds
        backends = getattr(BackendManager, f"_{group}_backends", {})
        for backend_name, backend_impl in backends.items():
            try:
                manager = BackendManager(group, backend_impl)
                _DEFAULT_BACKENDS[group] = manager.backend
                break
            except ImportError as err:
                _IMPORT_ERRORS[f"{group}.{backend_name}"] = err

        if group not in _DEFAULT_BACKENDS:
            available = ", ".join(backends.keys()) if backends else "none registered"
            raise ImportError(
                f"No backend could be loaded for group '{group}'. "
                f"Available backends: {available}"
            )

    backend = _DEFAULT_BACKENDS[group]
    if submodule is not None:
        if not submodule.startswith("."):
            submodule = "." + submodule
        return importlib.import_module(submodule, package=backend.__name__)

    return backend


# For backward compatibility
class set_gym_backend(BackendManager):  # noqa: N801
    """Legacy interface for gym backend selection."""
    def __init__(self, backend: str | Any | Callable[[], Any]):
        super().__init__("gym", backend)

def gym_backend(submodule: str | None = None) -> Any:
    """Legacy interface for gym backend access."""
    return get_backend("gym", submodule)

register_backend("gym", {
    "gym": "gym",
    "gymnasium": "gymnasium",
})
