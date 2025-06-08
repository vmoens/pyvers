# Dynamic dispatch implementation based on module versions.
from __future__ import annotations

import collections
import inspect
import logging
import sys
import warnings
from collections.abc import Callable
from copy import copy
from functools import wraps
from importlib import import_module
from typing import Any, TypeVar

from packaging.version import parse

logger = logging.getLogger(__name__)

# Global flag for verbose output
VERBOSE = False

T = TypeVar("T", bound=Callable)

class implement_for:  # noqa: N801
    """A version decorator that checks version compatibility and implements functions.

    If specified module is missing or there is no fitting implementation, call of
    the decorated function will lead to the explicit error.
    In case of intersected ranges, last fitting implementation is used.

    This wrapper also works to implement different backends for a same function
    (eg. gym vs gymnasium, numpy vs jax-numpy etc).

    Args:
        module_name (str or callable): version is checked for the module with this
            name (e.g. "gym"). If a callable is provided, it should return the
            module.
        from_version: version from which implementation is compatible.
            Can be open (None).
        to_version: version from which implementation is no longer compatible.
            Can be open (None).

    Keyword Args:
        class_method (bool, optional): if ``True``, the function will be written
            as a class method. Defaults to ``False``.
        compilable (bool, optional): If ``False``, the module import happens
            only on the first call to the wrapped function. If ``True``, the
            module import happens when the wrapped function is initialized.
            Defaults to ``False``.

    Examples:
        >>> @implement_for("gym", "0.13", "0.14")
        >>> def fun(self, x):
        ...     # Older gym versions will return x + 1
        ...     return x + 1
        ...
        >>> @implement_for("gym", "0.14", "0.23")
        >>> def fun(self, x):
        ...     # More recent gym versions will return x + 2
        ...     return x + 2
        ...
        >>> @implement_for(lambda: import_module("gym"), "0.23", None)
        >>> def fun(self, x):
        ...     # More recent gym versions will return x + 2
        ...     return x + 2
        ...
        >>> @implement_for("gymnasium", None, "1.0.0")
        >>> def fun(self, x):
        ...     # If gymnasium is to be used instead of gym, x+3 will be returned
        ...     return x + 3
        ...

        This indicates that the function is compatible with gym 0.13+,
        but doesn't with gym 0.14+.
    """

    # Stores pointers to fitting implementations: dict[func_name] = func_pointer
    _implementations: dict[str, implement_for] = {}
    _setters: list[implement_for] = []
    _cache_modules: dict[str, Any] = {}

    def __init__(
        self,
        module_name: str | Callable[[], Any],
        from_version: str | None = None,
        to_version: str | None = None,
        *,
        class_method: bool = False,
        compilable: bool = False,
    ):
        self.module_name = module_name
        self.from_version = from_version
        self.to_version = to_version
        self.class_method = class_method
        self._compilable = compilable
        self.fn: Callable | None = None
        self.func_name: str | None = None
        self.do_set: bool = False
        implement_for._setters.append(self)

    @staticmethod
    def check_version(
        version: str, from_version: str | None, to_version: str | None
    ) -> bool:
        version = parse(".".join([str(v) for v in parse(version).release]))
        return (from_version is None or version >= parse(from_version)) and (
            to_version is None or version < parse(to_version)
        )

    @staticmethod
    def get_class_that_defined_method(f: Callable) -> Any | None:
        """Returns the class of a method, if it is defined, and None otherwise."""
        out = f.__globals__.get(f.__qualname__.split(".")[0], None)
        return out

    @classmethod
    def get_func_name(cls, fn: Callable) -> str:
        # produces a name like module.Class.method or module.function
        fn_str = str(fn).split(".")
        if fn_str[0].startswith("<bound method "):
            first = fn_str[0][len("<bound method ") :]
        elif fn_str[0].startswith("<function "):
            first = fn_str[0][len("<function ") :]
        else:
            raise RuntimeError(f"Unknown func representation {fn}")
        last = fn_str[1:]
        if last:
            first = [first]
            last[-1] = last[-1].split(" ")[0]
        else:
            last = [first.split(" ")[0]]
            first = []
        return ".".join([fn.__module__] + first + last)

    def _get_cls(self, fn: Callable) -> Any | None:
        cls = self.get_class_that_defined_method(fn)
        if cls is None:
            # class not yet defined
            return None
        if cls.__class__.__name__ == "function":
            cls = inspect.getmodule(fn)
        return cls

    def module_set(self) -> None:
        """Sets the function in its module, if it exists already."""
        if self.fn is None:
            return
        prev_setter = type(self)._implementations.get(self.get_func_name(self.fn), None)
        if prev_setter is not None:
            prev_setter.do_set = False
        type(self)._implementations[self.get_func_name(self.fn)] = self
        cls = self.get_class_that_defined_method(self.fn)
        if cls is not None:
            if cls.__class__.__name__ == "function":
                cls = inspect.getmodule(self.fn)
        else:
            # class not yet defined
            return
        try:
            delattr(cls, self.fn.__name__)
        except AttributeError:
            pass

        name = self.fn.__name__
        if self.class_method:
            fn = classmethod(self.fn)
        else:
            fn = self.fn
        setattr(cls, name, fn)

    @classmethod
    def import_module(cls, module_name: str | Callable[[], Any]) -> str:
        """Imports module and returns its version."""
        if not callable(module_name):
            module = cls._cache_modules.get(module_name, None)
            if module is None:
                if module_name in sys.modules:
                    sys.modules[module_name] = module = import_module(module_name)
                else:
                    cls._cache_modules[module_name] = module = import_module(
                        module_name
                    )
        else:
            module = module_name()
        return module.__version__

    _lazy_impl = collections.defaultdict(list)

    def _delazify(self, func_name: str) -> Callable | None:
        out = None
        for local_call in implement_for._lazy_impl[func_name]:
            out = local_call()
        return out

    def __call__(self, fn: T) -> T:
        # function names are unique
        self.func_name = self.get_func_name(fn)
        self.fn = fn
        implement_for._lazy_impl[self.func_name].append(self._call)

        if self._compilable:
            _call_fn = self._delazify(self.func_name)

            if self.class_method and _call_fn is not None:
                return classmethod(_call_fn)  # type: ignore

            return _call_fn if _call_fn is not None else fn

        @wraps(fn)
        def _lazy_call_fn(*args: Any, **kwargs: Any) -> Any:
            # first time we call the function, we also do the replacement.
            # This will cause the imports to occur only during the first call to fn
            result = self._delazify(self.func_name)
            if result is not None:
                return result(*args, **kwargs)
            return fn(*args, **kwargs)

        if self.class_method:
            return classmethod(_lazy_call_fn)  # type: ignore

        return _lazy_call_fn  # type: ignore

    def _check_backend_conflict(self, version: str, func_name: str) -> bool:
        """Check if there's a backend conflict and handle it."""
        if self.check_version(version, self.from_version, self.to_version):
            if VERBOSE:
                module = (
                    import_module(self.module_name)
                    if isinstance(self.module_name, str)
                    else self.module_name()
                )
                msg = (
                    f"Got multiple backends for {func_name}. "
                    f"Using last queried ({module}, version {version})."
                )
                warnings.warn(msg, stacklevel=2)
            return True
        return False

    def _handle_existing_implementation(
        self, func_name: str, implementations: dict[str, implement_for]
    ) -> Callable | None:
        """Handle the case where an implementation already exists."""
        try:
            version = self.import_module(self.module_name)
            if self._check_backend_conflict(version, func_name):
                self.do_set = True
            if not self.do_set:
                return implementations[func_name].fn
        except ModuleNotFoundError:
            return implementations[func_name].fn
        return None

    def _handle_new_implementation(self) -> bool:
        """Handle the case where this is a new implementation."""
        try:
            version = self.import_module(self.module_name)
            return self.check_version(version, self.from_version, self.to_version)
        except ModuleNotFoundError:
            return False

    def _call(self) -> Callable:
        """Handle the function call and return appropriate implementation."""
        if self.fn is None:
            raise RuntimeError("Function not set")

        fn = self.fn
        func_name = self.func_name
        if func_name is None:
            raise RuntimeError("Function name not set")

        implementations = implement_for._implementations

        @wraps(fn)
        def unsupported(*args: Any, **kwargs: Any) -> Any:
            raise ModuleNotFoundError(
                f"Supported version of '{func_name}' has not been found."
            )

        self.do_set = False
        if func_name in implementations:
            result = self._handle_existing_implementation(func_name, implementations)
            if result is not None:
                return result
        else:
            self.do_set = self._handle_new_implementation()
            if not self.do_set:
                return unsupported

        if self.do_set:
            self.module_set()
            return fn
        return unsupported

    @classmethod
    def reset(cls, setters_dict: dict[str, implement_for] | None = None) -> None:
        """Resets the setters in setter_dict.

        Args:
            setters_dict: A copy of implementations. We iterate through its values
                and call :meth:`module_set` for each.
        """
        if VERBOSE:
            logger.info("resetting implement_for")
        if setters_dict is None:
            setters_dict = copy(cls._implementations)
        for setter in setters_dict.values():
            setter.module_set()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(module_name={self.module_name}, "
            f"from_version={self.from_version}, to_version={self.to_version})"
        )
