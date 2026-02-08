"""
Example: Using device detection in baseline training.

Shows how to integrate device.py in your training scripts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.device import get_device, move_to_device, set_reproducible, print_device_info


class SimpleModel(nn.Module):
    """Example model for demonstration."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_example():
    """Example training loop with device management."""

    # 1. Set reproducibility
    set_reproducible(seed=42)

    # 2. Get device (auto-detects CUDA or CPU)
    device = get_device()
    print_device_info(device)

    # 3. Create model and move to device
    model = SimpleModel(input_dim=10, hidden_dim=50, output_dim=2)
    model = move_to_device(model, device)

    # 4. Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 5. Example training loop
    print("Starting training...")
    for epoch in range(5):
        # Create dummy batch
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))

        # Move batch to device
        x, y = move_to_device([x, y], device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

    print("\n✓ Training completed!")


if __name__ == "__main__":
    train_example()
