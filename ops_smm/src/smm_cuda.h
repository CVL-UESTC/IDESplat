/*!
**************************************************************************************************
* Sparse Matrix Multiplication (SMM)
* Licensed under The MIT License [see LICENSE for details]
**************************************************************************************************
*/

#pragma once
#include <torch/extension.h>

// 前向传播函数声明
at::Tensor SMM_QmK_forward_cuda(const at::Tensor &A, const at::Tensor &B, const at::Tensor &index);
at::Tensor SMM_AmV_forward_cuda(const at::Tensor &A, const at::Tensor &B, const at::Tensor &index);

// 反向传播函数声明
std::vector<at::Tensor> SMM_QmK_backward_cuda(const at::Tensor &grad_output, const at::Tensor &A, const at::Tensor &B, const at::Tensor &index);
std::vector<at::Tensor> SMM_AmV_backward_cuda(const at::Tensor &grad_output, const at::Tensor &A, const at::Tensor &B, const at::Tensor &index);