import torch

if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU.")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    x = torch.rand(3, 3).cuda()
    print("Tensor on GPU:", x)
else:
    print("CUDA is not available. PyTorch cannot use the GPU.")
