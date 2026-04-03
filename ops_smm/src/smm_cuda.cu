#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <vector>


///////////// SMM_QmK

// CUDA kernel for forward propagation
__global__ void SMM_QmK_forward_kernel(const float* A, const float* B, const int* index, float* C, int Batch, int N, int K, int C_dim, int B_cols) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Corresponds to N
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Corresponds to K

    if (row < N && col < K) {
        int idx = batch * N * K + row * K + col;
        int b_col = index[idx];
        
        // 检查索引有效性
        if (b_col < 0) {
            C[idx] = 0.0f;
            return;
        }
        
        float value = 0.0f;
        for (int e = 0; e < C_dim; ++e) {
            int a_idx = batch * N * C_dim + row * C_dim + e;
            int b_idx = batch * C_dim * B_cols + e * B_cols + b_col;
            value += A[a_idx] * B[b_idx];
        }
        C[idx] = value;
    }
}



// Forward propagation function
at::Tensor SMM_QmK_forward_cuda(const at::Tensor &A, const at::Tensor &B, const at::Tensor &index) {

    // Check if tensors are contiguous
    AT_ASSERTM(A.is_contiguous(), "A tensor must be contiguous");
    AT_ASSERTM(B.is_contiguous(), "B tensor must be contiguous");
    AT_ASSERTM(index.is_contiguous(), "Index tensor must be contiguous");
    const int Batch = A.size(0);
    const int N = A.size(1);   // Dimension N of A
    const int C_dim = A.size(2);  // Dimension C of A (which is the row count of B)
    const int K = index.size(2);
    const int B_cols = B.size(2);  // Column count of B

    auto C = at::zeros({Batch, N, K}, A.options().dtype(torch::kFloat32));

    const int threads =16;
    const dim3 block_dim(threads, threads);
    const dim3 grid_dim((K + threads - 1) / threads, (N + threads - 1) / threads, Batch);

    SMM_QmK_forward_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), index.data_ptr<int>(), C.data_ptr<float>(), Batch, N, K, C_dim, B_cols
    );

    return C;
}



// 独立计算grad_A的核函数
__global__ void SMM_QmK_backward_gradA_kernel(
    const float* grad_output,
    const float* B_T,
    const int* index,
    float* grad_A,
    int Batch, int N, int K, int C_dim, int B_cols)
{
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < Batch && row < N && col < C_dim) {
        float grad_value = 0.0f;
        for (int k = 0; k < K; ++k) {
            int b_row = index[batch * N * K + row * K + k];
            if (b_row < 0) {
                continue;
            }
            else {
                grad_value += grad_output[batch * N * K + row * K + k] * B_T[batch * B_cols * C_dim + b_row * C_dim + col];
            }
        }
        grad_A[batch * N * C_dim + row * C_dim + col] = grad_value;
    }
}


// 独立计算grad_B的核函数
__global__ void SMM_QmK_backward_gradB_kernel(
    const float* grad_output,
    const float* A_T,
    const int* index,
    float* grad_B,
    int Batch, int N, int K, int C_dim, int B_cols)
{
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y; // C_dim
    int col = blockIdx.x * blockDim.x + threadIdx.x; // K

    if (batch < Batch && row < C_dim && col < K) {
        for (int n = 0; n < N; ++n) {
            int b_col = index[batch * N * K + n * K + col];
            if (b_col < 0) {
                continue;
            }
            else {
                float a_val = A_T[batch * C_dim * N + row * N + n]; // A_T shape: (Batch, C, N)
                float g = grad_output[batch * N * K + n * K + col];
                atomicAdd(&grad_B[batch * C_dim * B_cols + row * B_cols + b_col], a_val * g);
                // grad_B[batch * C_dim * B_cols + row * B_cols + b_col] += a_val * g;
            }
            
            
        }
    }
}


std::vector<at::Tensor> SMM_QmK_backward_cuda(const at::Tensor &grad_output,
                                                       const at::Tensor &A,
                                                       const at::Tensor &B,
                                                       const at::Tensor &index) {
    // Check the contiguity and device of the inputs
    AT_ASSERTM(A.is_contiguous(), "A tensor has to be contiguous");
    AT_ASSERTM(B.is_contiguous(), "B tensor has to be contiguous");
    AT_ASSERTM(index.is_contiguous(), "index tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    // Get dimensions of A and B
    const int Batch = A.size(0);
    const int N = A.size(1);    // Corresponds to dimension N of A
    const int C_dim = A.size(2); // Corresponds to dimension C of A
    const int K = index.size(2); // Corresponds to dimension K of index
    const int B_cols = B.size(2); // Corresponds to the column count of B

    // Allocate gradient tensors
    auto grad_A = at::zeros_like(A);
    auto grad_B = at::zeros_like(B);
    auto A_T = A.transpose(1, 2).contiguous(); // A^T (dimension swap)
    auto B_T = B.transpose(1, 2).contiguous(); // B^T (dimension swap)

    // 独立配置两个核函数的执行参数
    // grad_A核函数配置（N x C_dim网格）
    const int threads =16;
    dim3 grid_gradA((C_dim + threads-1)/threads, (N + threads-1)/threads, Batch);
    dim3 block_gradA(threads, threads);
    
    // grad_B核函数配置（B_cols x C_dim网格）
    dim3 grid_gradB((K + threads-1)/threads, (C_dim + threads-1)/threads, Batch);
    dim3 block_gradB(threads, threads);

    // 分别启动核函数
    SMM_QmK_backward_gradA_kernel<<<grid_gradA, block_gradA>>>(
        grad_output.data_ptr<float>(),
        B_T.data_ptr<float>(),
        index.data_ptr<int>(),
        grad_A.data_ptr<float>(),
        Batch, N, K, C_dim, B_cols
    );

    SMM_QmK_backward_gradB_kernel<<<grid_gradB, block_gradB>>>(
        grad_output.data_ptr<float>(),
        A_T.data_ptr<float>(),
        index.data_ptr<int>(),
        grad_B.data_ptr<float>(),
        Batch, N, K, C_dim, B_cols
    );

    return {grad_A, grad_B};
}





///////////// SMM_AmV

// CUDA kernel for forward propagation
__global__ void SMM_AmV_forward_kernel(const float* A, const float* B, const int* index, float* C, int Batch, int N, int K, int M, int C_dim) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < C_dim) {
        float value = 0.0;
        for (int k = 0; k < K; ++k) {
            int b_row = index[batch * N * K + row * K + k];
            if (b_row < 0) {
                continue;
            }
            else {
                value += A[batch * N * K + row * K + k] * B[batch * M * C_dim + b_row * C_dim + col];
            }
            
        }
        C[batch * N * C_dim + row * C_dim + col] = value;
    }
}


// Forward propagation function
at::Tensor SMM_AmV_forward_cuda(const at::Tensor &A, const at::Tensor &B, const at::Tensor &index) {
    // Ensure the tensors are contiguous and on the correct device
    AT_ASSERTM(A.is_contiguous(), "A tensor must be contiguous");
    AT_ASSERTM(B.is_contiguous(), "B tensor must be contiguous");
    AT_ASSERTM(index.is_contiguous(), "Index tensor must be contiguous");

    const int Batch = A.size(0);
    const int N = A.size(1);
    const int K = A.size(2);
    const int M = B.size(1);  // Row count of B
    const int C_dim = B.size(2);
    

    auto C = at::zeros({Batch, N, C_dim}, A.options().dtype(torch::kFloat32));

    const int threads =16;
    const dim3 block_dim(threads, threads);
    const dim3 grid_dim((C_dim + threads - 1) / threads, (N + threads - 1) / threads, Batch);

    SMM_AmV_forward_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), index.data_ptr<int>(), C.data_ptr<float>(), Batch, N, K, M, C_dim
    );

    return C;
}

// 独立计算grad_A的核函数（聚焦M维和K维）
__global__ void SMM_AmV_backward_gradA_kernel(
    const float* grad_output,
    const float* B_T,
    const int* index,
    float* grad_A,
    int Batch, int N, int K, int M, int C_dim)
{
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M维度
    int col = blockIdx.x * blockDim.x + threadIdx.x;     // K维度

    if (batch < Batch && row < N && col < K) {
        int idx = batch * N * K + row * K + col;
        float grad_value = 0.0f;
        int b_col = index[idx];
        if (b_col < 0) {
            grad_A[idx] = 0.0f;
            return;
        }
        else {
            for (int e = 0; e < C_dim; ++e) {
                grad_value += grad_output[batch * N * C_dim + row * C_dim + e] * B_T[batch * C_dim * M + e * M + b_col];
            }
        }
        grad_A[idx] = grad_value; // 直接写入，无需原子操作
    }
}


// 修改后的grad_B核函数
__global__ void SMM_AmV_backward_gradB_kernel(
    const float* grad_output,
    const float* A_T,
    const int* index_T,
    float* grad_B,
    int Batch, int N, int K, int M, int C_dim)
{
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < Batch && row < K && col < C_dim) {
        for (int n = 0; n < N; ++n) {
            int b_row = index_T[batch * K * N + row * N + n];
            if (b_row < 0) {
                continue;
            }
            else {
                float a_val = A_T[batch * K * N + row * N + n];
                float g = grad_output[batch * N * C_dim + n * C_dim + col];
                atomicAdd(&grad_B[batch * N * C_dim + b_row * C_dim + col], a_val * g);
                // grad_B[batch * M * C_dim + b_row * C_dim + col] += a_val * g;
            }
        }
    }
}


// Backward propagation function
std::vector<at::Tensor> SMM_AmV_backward_cuda(const at::Tensor &grad_output, const at::Tensor &A, const at::Tensor &B, const at::Tensor &index) {
    // Ensure tensors are contiguous and on the correct device
    AT_ASSERTM(A.is_contiguous(), "A tensor has to be contiguous");
    AT_ASSERTM(B.is_contiguous(), "B tensor has to be contiguous");
    AT_ASSERTM(index.is_contiguous(), "Index tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    const int Batch = A.size(0);
    const int N = A.size(1);
    const int K = A.size(2);
    const int M = B.size(1);  // Row count of B
    const int C_dim = B.size(2); 

    auto grad_A = at::zeros_like(A);
    auto grad_B = at::zeros_like(B);
    auto A_T = A.transpose(1, 2).contiguous(); // A^T (dimension swap)
    auto B_T = B.transpose(1, 2).contiguous(); // B^T (dimension swap)
    auto index_T = index.transpose(1, 2).contiguous();

    // 重新配置执行参数
    const int threads =16;
    
    dim3 grid_gradA((K + threads-1)/threads, (N + threads-1)/threads, Batch);
    dim3 block_gradA(threads, threads);
    
    dim3 grid_gradB((C_dim + threads-1)/threads, (K + threads-1)/threads, Batch);
    dim3 block_gradB(threads, threads);

    // 分别启动核函数
    SMM_AmV_backward_gradA_kernel<<<grid_gradA, block_gradA>>>(
        grad_output.data_ptr<float>(), B_T.data_ptr<float>(), 
        index.data_ptr<int>(), grad_A.data_ptr<float>(), 
        Batch, N, K, M, C_dim
    );

    SMM_AmV_backward_gradB_kernel<<<grid_gradB, block_gradB>>>(
        grad_output.data_ptr<float>(), A_T.data_ptr<float>(), 
        index_T.data_ptr<int>(), grad_B.data_ptr<float>(), 
        Batch, N, K, M, C_dim
    );
    return {grad_A, grad_B};
}



// Module registration
PYBIND11_MODULE(smm_cuda, m) {
    m.def("SMM_QmK_forward_cuda", &SMM_QmK_forward_cuda, "Sparse Matrix Multiplication Forward for Q @ K (CUDA)");
    m.def("SMM_QmK_backward_cuda", &SMM_QmK_backward_cuda, "Sparse Matrix Multiplication Backward for Q @ K (CUDA)");

    m.def("SMM_AmV_forward_cuda", &SMM_AmV_forward_cuda, "Sparse Matrix Multiplication Forward for A @ V  (CUDA)");
    m.def("SMM_AmV_backward_cuda", &SMM_AmV_backward_cuda, "Sparse Matrix Multiplication Backward for A @ V(CUDA)");
}
