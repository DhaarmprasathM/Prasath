import torch
import time

device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))

# Create large matrices
a = torch.randn(5000, 5000, device=device)
b = torch.randn(5000, 5000, device=device)

torch.cuda.synchronize()
start = time.time()

c = torch.matmul(a, b)

torch.cuda.synchronize()
end = time.time()

print("Matrix multiplication completed")
print("Output device:", c.device)
print("Execution time:", end - start, "seconds")
print("GPU Memory Allocated:",
      torch.cuda.memory_allocated() / 1024**3, "GB")