"""Importable fixtures for ``implement_for`` pickle tests."""

from pyvers import implement_for

__version__ = "0.3"


@implement_for("_pickle_utils", "0.3")
def before_dispatch(value):
    return f"before:{value}"


@implement_for("_pickle_utils", None, "0.3")
def repeated_decorator(value):
    return f"old:{value}"


@implement_for("_pickle_utils", "0.3")
def repeated_decorator(value):  # noqa: F811
    return f"selected:{value}"


@implement_for("_pickle_utils")
def registered(value):
    raise NotImplementedError("No matching implementation")


@registered.register(from_version="0.3")
def _(value):
    return f"registered:{value}"


@implement_for("_pickle_utils", "0.3")
def fresh_process(value):
    return f"fresh:{value}"


@implement_for("_pickle_utils", "0.3")
def copy_target(value):
    return f"copy:{value}"


@implement_for("_pickle_utils", "0.4")
def unsupported_version():
    return "unsupported"


@implement_for("pyvers_missing_pickle_dependency")
def missing_module():
    return "missing"


class PickleableMethods:
    @implement_for("_pickle_utils", "0.3")
    def instance_method(self, value):
        return f"method:{value}"
