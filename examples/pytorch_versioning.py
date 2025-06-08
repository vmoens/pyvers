# WARNING: This example requires two separate environments:
#  - One with torch < 2.0 (e.g. torch 1.13.1)
#  - One with torch >= 2.0 (e.g. torch 2.0.0)
# The example demonstrates how to use torch.compile optimization
# which was introduced in PyTorch 2.0

import torch
import torch.nn as nn

from pyvers import get_backend, implement_for, register_backend

# Register pytorch backend versions
register_backend(group="torch", backends={"torch": "torch"})

class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# For PyTorch < 2.0, we just return the model as is
@implement_for("torch", from_version=None, to_version="2.0.0")
def make_net(input_size=784, hidden_size=128, output_size=10):
    print(f"Using PyTorch {get_backend('torch').__version__} (pre-2.0)")
    model = SimpleNet(input_size, hidden_size, output_size)
    return model

# For PyTorch >= 2.0, we can use torch.compile for better performance
@implement_for("torch", from_version="2.0.0")
def make_net(input_size=784, hidden_size=128, output_size=10):  # noqa: F811
    print(f"Using PyTorch {get_backend('torch').__version__} (2.0+)")
    torch = get_backend("torch")
    model = SimpleNet(input_size, hidden_size, output_size)
    # Use torch.compile to optimize the model
    return torch.compile(model)

if __name__ == "__main__":
    # Create a small test input
    batch_size = 32
    input_size = 784  # e.g., flattened MNIST images
    x = torch.randn(batch_size, input_size)

    # Create the model - it will automatically use the right version
    model = make_net()

    # Run a forward pass
    print("\nRunning forward pass...")
    output = model(x)
    print(f"Output shape: {output.shape}")

    print("\n" + "="*80)
    if float(torch.__version__.split('.')[0]) >= 2:
        print("You're running this with PyTorch 2+.")
        print("The model was automatically optimized using torch.compile!")
        print("Try creating a new environment with PyTorch < 2.0 (e.g. 1.13.1) and run this again!")
    else:
        print("You're running this with PyTorch 1.x.")
        print("Try creating a new environment with PyTorch >= 2.0 to see the model optimized with torch.compile!")
