# Device Management Utilities

Automatic detection and management of CUDA GPU vs CPU for PyTorch models.

## Features

✅ **Auto-detection**: Automatically detects CUDA GPU availability  
✅ **Fallback**: Seamlessly falls back to CPU if no GPU available  
✅ **Multi-GPU**: Support for multiple GPUs with device ID selection  
✅ **Manual override**: Force CPU mode when needed  
✅ **Environment variable**: Set `FORCE_CPU=1` to force CPU  
✅ **Reproducibility**: Set random seeds across all frameworks  
✅ **Device info**: Get detailed device information  

## Quick Start

```python
from src.utils.device import get_device, move_to_device

# Auto-detect device (CUDA or CPU)
device = get_device()

# Move model to device
model = MyModel()
model = move_to_device(model, device)

# Move tensors to device
x = torch.randn(10, 5)
x = move_to_device(x, device)
```

## Usage Examples

### 1. Basic Auto-Detection

```python
from src.utils.device import get_device

# Automatically detects CUDA or falls back to CPU
device = get_device()
print(device)  # cuda:0 or cpu
```

### 2. Force CPU Mode

```python
# Method 1: Function argument
device = get_device(force_cpu=True)

# Method 2: Environment variable
import os
os.environ['FORCE_CPU'] = '1'
device = get_device()
```

### 3. Select Specific GPU

```python
# Use GPU 1 instead of GPU 0
device = get_device(device_id=1)
```

### 4. Get Device Information

```python
from src.utils.device import print_device_info

# Print detailed info
print_device_info()
```

**Output example (GPU):**
```
============================================================
Device Information
============================================================
  Device.................................. cuda:0
  Type.................................... cuda
  Is Cuda................................. True
  Gpu Id.................................. 0
  Gpu Name................................ NVIDIA GeForce RTX 3090
  Cuda Version............................ 11.8
  Total Memory Gb......................... 24.0
  Available Gpus.......................... 2
  Cudnn Enabled........................... True
  Cudnn Version........................... 8902
============================================================
```

**Output example (CPU):**
```
============================================================
Device Information
============================================================
  Device.................................. cpu
  Type.................................... cpu
  Is Cuda................................. False
  Cpu Count............................... 8
  Num Threads............................. 4
============================================================
```

### 5. Move Objects to Device

```python
from src.utils.device import move_to_device

device = get_device()

# Move single tensor
x = torch.randn(10, 5)
x = move_to_device(x, device)

# Move list of tensors
tensors = [torch.randn(5, 5) for _ in range(3)]
tensors = move_to_device(tensors, device)

# Move dict of tensors
batch = {'input': torch.randn(32, 10), 'target': torch.randn(32, 1)}
batch = move_to_device(batch, device)

# Move model
model = MyModel()
model = move_to_device(model, device)
```

### 6. Set Reproducibility

```python
from src.utils.device import set_reproducible

# Set random seeds for reproducibility
set_reproducible(seed=42)

# Now all random operations are deterministic:
# - Python random
# - NumPy random
# - PyTorch CPU random
# - PyTorch CUDA random
# - CuDNN operations
```

### 7. Training Loop Integration

```python
import torch
import torch.nn as nn
from src.utils.device import get_device, move_to_device, set_reproducible

# Setup
set_reproducible(42)
device = get_device()
model = move_to_device(MyModel(), device)
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:
        # Move batch to device
        batch_data = move_to_device(batch_data, device)
        batch_labels = move_to_device(batch_labels, device)
        
        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## API Reference

### `get_device(force_cpu=False, device_id=None)`
Get PyTorch device with automatic CUDA detection.

**Args:**
- `force_cpu` (bool): Force CPU usage even if CUDA available
- `device_id` (int): Specific GPU ID to use (default: 0)

**Returns:**
- `torch.device`: Device object

### `get_device_info(device=None)`
Get detailed device information.

**Args:**
- `device` (torch.device): Device to query (default: auto-detected)

**Returns:**
- `dict`: Device information

### `move_to_device(obj, device=None)`
Move tensor/model/collection to device.

**Args:**
- `obj`: Tensor, model, list, tuple, or dict
- `device` (torch.device): Target device (default: auto-detected)

**Returns:**
- Object moved to device

### `set_reproducible(seed=42)`
Set random seeds for reproducibility.

**Args:**
- `seed` (int): Random seed value

### `print_device_info(device=None)`
Print formatted device information.

**Args:**
- `device` (torch.device): Device to query (default: auto-detected)

## Environment Variables

- `FORCE_CPU`: Set to `1`, `true`, or `yes` to force CPU usage

## Caching

Device detection is cached after first call for performance. To reset cache:

```python
from src.utils.device import clear_device_cache

clear_device_cache()
device = get_device()  # Re-detects device
```

## Multi-GPU Notes

When multiple GPUs are available:

1. **Default behavior**: Uses GPU 0
2. **Specific GPU**: Use `device_id` parameter
3. **DataParallel**: Wrap model after moving to device
4. **DistributedDataParallel**: Set device per process

```python
# Example: Use GPU 2
device = get_device(device_id=2)
model = move_to_device(model, device)
```

## Troubleshooting

### CUDA not detected even though GPU is available

Check:
1. PyTorch CUDA version: `torch.version.cuda`
2. CUDA installed: `nvidia-smi`
3. PyTorch built with CUDA: `torch.cuda.is_available()`

### Out of memory errors

```python
# Use CPU for large models
device = get_device(force_cpu=True)

# Or clear GPU cache
torch.cuda.empty_cache()
```

## Testing

Run the test script:

```bash
python test_device.py
```

Run the training example:

```bash
python example_device_usage.py
```

## Notes

- Works seamlessly on both GPU and CPU machines
- No code changes needed when moving between machines
- Colleague with GPU will automatically use it
- You on CPU will automatically fall back to CPU
- Both can run the same code without modifications!

## Integration Checklist

When adding to a new training script:

```python
# 1. Import utilities
from src.utils.device import get_device, move_to_device, set_reproducible

# 2. Set reproducibility (optional)
set_reproducible(seed=42)

# 3. Get device
device = get_device()

# 4. Move model to device
model = move_to_device(model, device)

# 5. Move batches in training loop
batch = move_to_device(batch, device)
```

That's it! Your code now works on both GPU and CPU automatically. 🎉
