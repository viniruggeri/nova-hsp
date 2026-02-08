"""
Test and demonstrate device detection utilities.
"""

from src.utils.device import (
    get_device,
    get_device_info,
    print_device_info,
    move_to_device,
    set_reproducible,
)
import torch

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Device Detection Test")
    print("=" * 80 + "\n")

    # 1. Auto-detect device
    print("1. Auto-detecting device...")
    device = get_device()
    print(f"   Detected: {device}\n")

    # 2. Print detailed info
    print("2. Detailed device information:")
    print_device_info()

    # 3. Test tensor creation
    print("3. Creating test tensor...")
    x = torch.randn(3, 3)
    x = move_to_device(x)
    print(f"   Tensor device: {x.device}")
    print(f"   Tensor shape: {x.shape}\n")

    # 4. Test with forced CPU
    print("4. Testing forced CPU mode...")
    cpu_device = get_device(force_cpu=True)
    print(f"   Device: {cpu_device}\n")

    # 5. Test model movement
    print("5. Creating and moving simple model...")
    model = torch.nn.Linear(10, 5)
    model = move_to_device(model)
    print(f"   Model device: {next(model.parameters()).device}\n")

    # 6. Set reproducibility
    print("6. Setting reproducible mode...")
    set_reproducible(seed=42)
    print("   ✓ Random seeds set for reproducibility\n")

    print("=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80 + "\n")
