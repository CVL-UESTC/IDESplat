import torch
from torch.autograd import Function
import smm_cuda  # 假设你的模块已经成功编译并导入

# 使用固定整数数据代替随机数据，方便核对结果
A = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]], device='cuda', dtype=torch.float32, requires_grad=True)  # (Batch=1, N=5, C=3)
B = torch.tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]], device='cuda', dtype=torch.float32, requires_grad=True)  # (Batch=1, C=3, N=5)
index = torch.tensor([[[0, 1], [1, 0], [2, 2], [0, 1], [1, 0]]], device='cuda', dtype=torch.int32)  # (Batch=1, N=5, k=2)

# 定义自定义的选择性矩阵乘法操作
class SparseMM(Function):
    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        return smm_cuda.SMM_QmK_forward_cuda(A, B, index)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        print("SparseMM grad_output:", grad_output)
        A, B, index = ctx.saved_tensors
        # 确保 grad_output 是连续的
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_A, grad_B = smm_cuda.SMM_QmK_backward_cuda(grad_output, A, B, index)
        return grad_A, grad_B, None


# 定义一个简单的神经网络层，用于测试前向和反向传播
class SparseMMLayer(torch.nn.Module):
    def forward(self, A, B, index):
        return SparseMM.apply(A, B, index)

# 实例化自定义层
layer = SparseMMLayer()

# 执行前向传播
C = layer(A, B, index)
print("CUDA C = sparse_MM(A, B):")
print(C)

# 执行反向传播，检查梯度计算
loss = C.sum()
loss.backward()

print("CUDA A's gradient:")
print(A.grad)
print("CUDA B's gradient:")
print(B.grad)

# 手动计算 C 的结果
def manual_sparse_mm(A, B, index):
    batch_size, N, C_dim = A.shape
    _, C_dim_B, B_cols = B.shape
    _, N, K = index.shape
    C_manual = torch.zeros((batch_size, N, K), device=A.device)

    for batch in range(batch_size):
        for row in range(N):
            for col in range(K):
                b_col = index[batch, row, col].item()
                value = 0.0
                for e in range(C_dim):
                    value += A[batch, row, e].item() * B[batch, e, b_col].item()
                C_manual[batch, row, col] = value
    return C_manual

C_manual = manual_sparse_mm(A.detach().cpu(), B.detach().cpu(), index.detach().cpu())
print("Manual C:")
print(C_manual.cuda())

# 手动计算 A 和 B 的梯度
def manual_backward(grad_output, A, B, index):
    batch_size, N, C_dim = A.shape
    _, C_dim_B, B_cols = B.shape
    _, N, K = index.shape

    grad_A_manual = torch.zeros_like(A)
    grad_B_manual = torch.zeros_like(B)

    for batch in range(batch_size):
        for row in range(N):
            for col in range(K):
                b_col = index[batch, row, col].item()
                grad_value = grad_output[batch, row, col].item()

                for e in range(C_dim):
                    if batch==0 and row==0 and col==0:
                        print(e, B[batch, e, b_col])
                    grad_A_manual[batch, row, e] += grad_value * B[batch, e, b_col].item()
                    grad_B_manual[batch, e, b_col] += grad_value * A[batch, row, e].item()
                    
    return grad_A_manual, grad_B_manual

# 使用 loss 的梯度进行手动反向传播
grad_output = torch.ones_like(C).detach().cpu()  # 因为 loss 是 C.sum()，grad_output 是全1
grad_A_manual, grad_B_manual = manual_backward(grad_output, A.detach().cpu(), B.detach().cpu(), index.detach().cpu())

print("Manual A's gradient:")
print(grad_A_manual.cuda())
print("Manual B's gradient:")
print(grad_B_manual.cuda())

# 对比 CUDA 计算的结果和手动计算的结果
print("\nComparison between CUDA and manual results:")
print("C difference (CUDA - Manual):")
print(C - C_manual.cuda())

print("A gradient difference (CUDA - Manual):")
print(A.grad - grad_A_manual.cuda())

print("B gradient difference (CUDA - Manual):")
print(B.grad - grad_B_manual.cuda())
