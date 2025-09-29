import torch
from torch import Tensor
from typing import Tuple

# Step 1: Define custom operators using torch.library API
@torch.library.custom_op("my_ops::batchnorm_forward", mutates_args=("running_mean", "running_var"))
def batchnorm_forward(
    input: Tensor,           # [N, C, H, W]
    gamma: Tensor,           # [C]
    beta: Tensor,            # [C]
    running_mean: Tensor,    # [C]
    running_var: Tensor,     # [C]
    training: bool,
    momentum: float,
    eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """forward pass of BatchNorm for 4D input [N, C, H, W]."""

    # Implement Here
    
    return output, save_mean, save_invstd


@torch.library.custom_op("my_ops::batchnorm_backward", mutates_args=())
def batchnorm_backward(
    grad_output: Tensor,     # [N, C, H, W]
    input: Tensor,           # [N, C, H, W]
    gamma: Tensor,           # [C]
    save_mean: Tensor,       # [C]
    save_invstd: Tensor      # [C]
) -> Tuple[Tensor, Tensor, Tensor]:
    """backward pass of BatchNorm for 4D input."""

    # Implement Here

    return grad_input, grad_gamma, grad_beta


# Step 2: Connect forward and backward with autograd
# This connects our custom forward/backward operators to PyTorch's 
# autograd system, allowing gradients to flow during backpropagation
class BatchNormCustom(torch.autograd.Function):
    """
    Custom Batch Normalization for 4D inputs [N, C, H, W].
    
    Bridges custom operators with PyTorch's autograd engine.
    - forward(): calls custom forward operator and saves context
    - backward(): calls custom backward operator using saved context

    Usage:
        output = BatchNormCustom.apply(input, gamma, beta, running_mean, running_var, training, momentum, eps)
    """
    @staticmethod
    def forward(ctx, input, gamma, beta, running_mean, running_var, training, momentum, eps):
        output, save_mean, save_invstd = torch.ops.my_ops.batchnorm_forward(
            input, gamma, beta, running_mean, running_var, training, momentum, eps
        )
        ctx.save_for_backward(input, gamma, save_mean, save_invstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma, save_mean, save_invstd = ctx.saved_tensors
        grad_input, grad_gamma, grad_beta = torch.ops.my_ops.batchnorm_backward(
            grad_output, input, gamma, save_mean, save_invstd
        )
        # Return gradients for all forward inputs (None for non-tensor args)
        return grad_input, grad_gamma, grad_beta, None, None, None, None, None