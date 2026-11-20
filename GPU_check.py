import torch

print("=" * 60)
print("PyTorch GPU Diagnostic")
print("=" * 60)

print("PyTorch version :", torch.__version__)
print("PyTorch path    :", torch.__file__)

print("torch.version type :", type(torch.version))
print("torch.version      :", torch.version)

print("CUDA version    :", torch.version.cuda)

print("CUDA available  :", torch.cuda.is_available())

if torch.cuda.is_available():


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count       :", torch.cuda.device_count())
    print("GPU name        :", torch.cuda.get_device_name(0))
    print("Current GPU     :", torch.cuda.current_device())

print("=" * 60)
