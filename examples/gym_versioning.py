# Requires gymnasium and gym to be installed:
#  !pip install gymnasium "gym<0.25"

from pyvers import get_backend, implement_for, register_backend, set_backend

# First, we register the backends: the backends must be a dictionary of the form {backend_name: module_name}
register_backend(group="gym", backends={"gymnasium": "gymnasium", "gym": "gym"})

# Then, we implement the function for the gymnasium backend
#  The 'step' method returns terminated, truncated and not a single boolean like gym used to do
@implement_for("gymnasium")
def process_step(env, action):
    print(f"Using gymnasium backend: {get_backend('gym')} within the gymnasium context manager")
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, reward, terminated, truncated, info

# Then, we implement the function for the gym backend. Old gym versions used to return a single boolean for done.
#  This change was made in gym 0.25, which is why we implement it for gym versions before 0.25.
@implement_for("gym", from_version=None, to_version="0.26.0")
def process_step(env, action):  # noqa: F811
    print(f"Using gym backend: {get_backend('gym')} within the gym context manager")
    obs, reward, done, info = env.step(action)
    return obs, reward, done, info


if __name__ == "__main__":
    # Check that we have installed gym and gymnasium
    import gym
    import gymnasium

    assert gym.version.VERSION < "0.25.0"
    assert gymnasium.__version__ >= "0.25.0"

    print(f"Using gym backend: {get_backend('gym')}")
    # Check that we use the gym backend
    with set_backend("gym", "gym"):
        print(f"Using gym backend: {get_backend('gym')} within the gym context manager")
        # We use get_backend to get the gym backend dynamically
        env = get_backend("gym").make("CartPole-v0").unwrapped
        env.reset()
        assert len(process_step(env, 0)) == 4

    print(f"Using gym backend: {get_backend('gym')}")
    # Check that we use the gymnasium backend
    with set_backend("gym", "gymnasium"):
        print(f"Using gym backend: {get_backend('gym')} within the gymnasium context manager")
        env = get_backend("gym").make("CartPole-v1").unwrapped
        env.reset()
        assert len(process_step(env, 0)) == 5

    print(f"Using gym backend: {get_backend('gym')}")
    # Check that we use the gym backend again
    with set_backend("gym", "gym"):
        print(f"Using gym backend: {get_backend('gym')} within the gym context manager")
        env = get_backend("gym").make("CartPole-v0").unwrapped
        env.reset()
        assert len(process_step(env, 0)) == 4
