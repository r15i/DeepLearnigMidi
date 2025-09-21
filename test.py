# File: save_test_tensor.py
import torch
import numpy as np
import os

print(f"--- Running Tensor Sanity Check ---")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

# 1. Create a tensor exactly like the one in your script
test_array = np.zeros((128, 16), dtype=np.uint8)
test_tensor = torch.from_numpy(test_array)

print(f"Tensor created with shape: {test_tensor.shape}")
print(f"Tensor created with dtype: {test_tensor.dtype}")  # Should be torch.uint8

# 2. Save it
file_path = "test_tensor.pt"
torch.save(test_tensor, file_path)
print(f"Tensor saved to '{file_path}'")

# 3. Get file size for verification
file_size_bytes = os.path.getsize(file_path)
print(f"File size on disk: {file_size_bytes} bytes")
print(f"--- Test Complete ---")
