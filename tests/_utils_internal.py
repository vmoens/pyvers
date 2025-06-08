# Specified for test_utils.py
__version__ = "0.3"


from pyvers import implement_for


def _set_gym_environments():
    ...


@implement_for("gym", None, "0.21.0")
def _set_gym_environments():  # noqa: F811
    ...

@implement_for("gym", "0.21.0", None)
def _set_gym_environments():  # noqa: F811
    ...

@implement_for("gymnasium", None, "1.0.0")
def _set_gym_environments():  # noqa: F811
    ...

@implement_for("gymnasium", "1.0.0", "1.1.0")
def _set_gym_environments():  # noqa: F811
    ...

@implement_for("gymnasium", "1.1.0")
def _set_gym_environments():  # noqa: F811
    ...
