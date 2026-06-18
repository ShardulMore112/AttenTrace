import torch
import sys

def verify_gpu():
    print("=" * 50)
    print("PYTORCH & CUDA INTEGRATION CHECK")
    print("=" * 50)
    
    # Check Python and PyTorch versions
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check CUDA Availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        cuda_version = torch.version.cuda
        
        print(f"CUDA Version: {cuda_version}")
        print(f"Current Device ID: {current_device}")
        print(f"Device Name: {device_name}")
        
        # Simple tensor operation sanity check to guarantee execution works on the GPU
        try:
            x = torch.rand(3, 3).to("cuda")
            y = torch.rand(3, 3).to("cuda")
            z = torch.matmul(x, y)
            print("Tensor Operation Status: Success (GPU math operations verified)")
        except Exception as e:
            print(f"Tensor Operation Status: Failed\nError: {e}")
            
    else:
        print("\n[WARNING]: PyTorch cannot see your GPU.")
        print("Please verify that:")
        print("1. Your NVIDIA drivers are up to date.")
        print("2. You installed the CUDA-enabled PyTorch build (using the pip extra-index-url).")
        
    print("=" * 50)

if __name__ == "__main__":
    verify_gpu()