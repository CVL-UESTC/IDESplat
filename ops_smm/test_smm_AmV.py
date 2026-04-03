import torch
from torch.autograd import Function
import smm_cuda  # Make sure to import the correct module name


# Input tensors for testing
# A's shape is (Batch=2, M=3, K=2)
A = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]], device='cuda', dtype=torch.float32, requires_grad=True)  # (2, 3, 2)

# B's shape is (Batch=2, M=3, B_cols=2)
B = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]], device='cuda', dtype=torch.float32, requires_grad=True)  # (2, 3, 2)

# Index's shape is (Batch=2, M=3, K=2)
index = torch.tensor([[[0, 1], [1, 2], [0, 2]], [[2, 1], [1, 0], [0, 2]]], device='cuda', dtype=torch.int32)  # (2, 3, 2)


# Define the custom sparse matrix multiplication function using PyTorch's autograd
class SparseMMFunction(Function):
    @staticmethod
    def forward(ctx, A, B, index):
        # Save tensors for backward pass
        ctx.save_for_backward(A, B, index)
        # Call the CUDA function for forward pass
        return smm_cuda.SMM_AmV_forward_cuda(A, B, index)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        # Load saved tensors
        A, B, index = ctx.saved_tensors

        # Ensure the grad_output is contiguous
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        # Call the CUDA function for the backward pass
        grad_A, grad_B = smm_cuda.SMM_AmV_backward_cuda(grad_output, A, B, index)
        return grad_A, grad_B, None


# Define a custom layer using the sparse matrix multiplication function
class SparseMMLayer(torch.nn.Module):
    def forward(self, A, B, index):
        return SparseMMFunction.apply(A, B, index)


# Instantiate the layer and perform forward pass
layer = SparseMMLayer()
C = layer(A, B, index)

print("CUDA C = sparse_mm(A, B):")
print(C)

# Perform backward pass
loss = C.sum()
loss.backward()

# Print gradients
print("CUDA A's gradient:")
print(A.grad)
print("CUDA B's gradient:")
print(B.grad)


# Manually compute the result for comparison
def manual_sparse_mm(A, B, index):
    batch_size, M, K = A.shape
    _, M, B_cols = B.shape
    _, M, K = index.shape
    C_manual = torch.zeros((batch_size, M, B_cols), device=A.device)

    for batch in range(batch_size):
        for row in range(M):
            for col in range(B_cols):
                value = 0.0
                for e in range(K):
                    b_row = index[batch, row, e].item()
                    value += A[batch, row, e].item() * B[batch, b_row, col].item()
                C_manual[batch, row, col] = value
    return C_manual


# Compare the manual result with CUDA result
C_manual = manual_sparse_mm(A.detach().cpu(), B.detach().cpu(), index.detach().cpu())
print("Manual C:")
print(C_manual.cuda())

print("\nComparison between CUDA and manual results:")
print("C difference (CUDA - Manual):")
print(C - C_manual.cuda())


# Manually compute gradients for comparison
def manual_backward(grad_output, A, B, index):
    batch_size, M, K = A.shape
    _, M, B_cols = B.shape
    _, M, K = index.shape

    grad_A_manual = torch.zeros_like(A)
    grad_B_manual = torch.zeros_like(B)

    for batch in range(batch_size):
        for row in range(M):
            for col in range(B_cols):
                grad_value = grad_output[batch, row, col].item()
                for e in range(K):
                    b_row = index[batch, row, e].item()
                    grad_A_manual[batch, row, e] += grad_value * B[batch, b_row, col].item()
                    grad_B_manual[batch, b_row, col] += grad_value * A[batch, row, e].item()

    return grad_A_manual, grad_B_manual


# Compute manual gradients using the loss gradient
grad_output = torch.ones_like(C).detach().cpu()  # Since loss is C.sum(), grad_output is all ones
grad_A_manual, grad_B_manual = manual_backward(grad_output, A.detach().cpu(), B.detach().cpu(), index.detach().cpu())

# Print the manually computed gradients
print("Manual A's gradient:")
print(grad_A_manual.cuda())
print("Manual B's gradient:")
print(grad_B_manual.cuda())

# Compare gradients between CUDA and manual
print("A gradient difference (CUDA - Manual):")
print(A.grad - grad_A_manual.cuda())

print("B gradient difference (CUDA - Manual):")
print(B.grad - grad_B_manual.cuda())
