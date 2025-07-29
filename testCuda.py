
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0)) # Prints the name of your first GPU
print(torch.version.cuda) # Prints the CUDA version PyTorch was built with


#def main():
#     print("Hello from deeplearnigmidi!")


#if __name__ == "__main__":
#    main()
