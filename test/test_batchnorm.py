import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_custom_ops_bn import BatchNormCustom


def test_custom_batchnorm():
    print("Testing Custom BatchNorm Implementation")
    print("=" * 40)

    torch.manual_seed(42)

    # test configuration 
    N, C, H, W = 32, 64, 56, 56 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Input shape: [{N}, {C}, {H}, {W}]")

    # Create input tensor [N, C, H, W]
    input_custom = torch.randn(N, C, H, W, device=device, requires_grad=True)
    input_pytorch = input_custom.clone().detach().requires_grad_(True)

    # Initialize parameters
    gamma = torch.ones(C, device=device, requires_grad=True)
    beta = torch.zeros(C, device=device, requires_grad=True)
    running_mean = torch.zeros(C, device=device)
    running_var = torch.ones(C, device=device)

    # PyTorch's BatchNorm
    bn_pytorch = nn.BatchNorm2d(C, eps=1e-5, momentum=0.1, device=device)
    bn_pytorch.weight.data = gamma.clone()
    bn_pytorch.bias.data = beta.clone()
    bn_pytorch.running_mean.data = running_mean.clone()
    bn_pytorch.running_var.data = running_var.clone()

    # Forward pass ************************************************************************************************************
    print("\n1. Forward Pass")
    output_custom = BatchNormCustom.apply(input_custom, gamma, beta, running_mean, running_var, True, 0.1, 1e-5)
    output_pytorch = bn_pytorch(input_pytorch)

    forward_diff = torch.abs(output_custom - output_pytorch).max().item()
    print(f"   Max difference: {forward_diff:.2e}")
    print(f"   ✓ Forward pass {'passed' if forward_diff < 1e-5 else 'failed'}")
    # **************************************************************************************************************************

    # Backward pass 
    grad_output = torch.randn_like(output_custom)

    output_custom.backward(grad_output)
    output_pytorch.backward(grad_output.clone())

    input_grad_diff = torch.abs(input_custom.grad - input_pytorch.grad).max().item()
    gamma_grad_diff = torch.abs(gamma.grad - bn_pytorch.weight.grad).max().item()
    beta_grad_diff = torch.abs(beta.grad - bn_pytorch.bias.grad).max().item()

    mean_diff = torch.abs(running_mean - bn_pytorch.running_mean).max().item()
    var_diff = torch.abs(running_var - bn_pytorch.running_var).max().item()


    # Overall result
    all_passed = all(diff < 1e-3 for diff in [forward_diff, input_grad_diff, gamma_grad_diff, beta_grad_diff, mean_diff, var_diff])
    print(f"\n✓ Test {'PASSED' if all_passed else 'FAILED'}")


def test_inference_mode():
    print("\n" + "=" * 40)
    print("Testing Inference Mode")

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # inference configuration 
    N, C, H, W = 8, 128, 28, 28  
    print(f"Inference input shape: [{N}, {C}, {H}, {W}]")

    # Create test data
    input_tensor = torch.randn(N, C, H, W, device=device)
    gamma = torch.ones(C, device=device)
    beta = torch.zeros(C, device=device)
    running_mean = torch.randn(C, device=device) * 0.1  
    running_var = torch.abs(torch.randn(C, device=device)) * 0.5 + 0.8  

    output_custom = BatchNormCustom.apply(input_tensor, gamma, beta, running_mean.clone(), running_var.clone(), False, 0.1, 1e-5)

    bn_pytorch = nn.BatchNorm2d(C, eps=1e-5, device=device)
    bn_pytorch.weight.data = gamma
    bn_pytorch.bias.data = beta
    bn_pytorch.running_mean.data = running_mean.clone()
    bn_pytorch.running_var.data = running_var.clone()
    bn_pytorch.eval()
    
    output_pytorch = bn_pytorch(input_tensor)

    diff = torch.abs(output_custom - output_pytorch).max().item()
    print(f"✓ Inference test {'PASSED' if diff < 1e-5 else 'FAILED'}")


if __name__ == "__main__":
    try:
        test_custom_batchnorm()
        test_inference_mode()
        print("\n" + "=" * 40)
        print("All tests completed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()