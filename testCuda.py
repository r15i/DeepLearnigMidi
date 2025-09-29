import torch

print(torch.__version__)
print(torch.cuda.is_available())



if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # Prints the name of your first GPU
    print(torch.version.cuda)  # Prints the CUDA version PyTorch was built with


# expected output


"""
2.7.1+cu128
True
NVIDIA GeForce GTX 850M
12.8
"""
