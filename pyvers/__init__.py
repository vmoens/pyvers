# Dynamic dispatch library for Python.

__version__ = "0.1.0"

from .backends import (
    BackendManager,
    get_backend,
    gym_backend,
    register_backend,
    set_backend,
    set_gym_backend,
)
from .implement_for import implement_for

__all__ = ["implement_for", "set_backend", "get_backend", "register_backend", "BackendManager", "set_gym_backend", "gym_backend"]
