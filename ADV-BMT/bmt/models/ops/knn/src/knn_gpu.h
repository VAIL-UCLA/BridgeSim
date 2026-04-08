// Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
// Published at NeurIPS 2022
// Written by Li Jiang, Shaoshuai Shi 
// All Rights Reserved


#ifndef KNN_H
#define KNN_H
#include <torch/serialize/tensor.h>
#include<vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <math.h>
// #include <THC/THC.h>


void knn_batch(at::Tensor xyz_tensor, at::Tensor query_xyz_tensor, at::Tensor batch_idxs_tensor, at::Tensor query_batch_offsets_tensor, at::Tensor idx_tensor, int n, int m, int k);
void knn_batch_cuda(int n, int m, int k, const float *xyz, const float *query_xyz, const int *batch_idxs, const int *query_batch_offsets, int *idx, cudaStream_t stream);

void knn_batch_mlogk(at::Tensor xyz_tensor, at::Tensor query_xyz_tensor, at::Tensor batch_idxs_tensor, at::Tensor query_batch_offsets_tensor, at::Tensor idx_tensor, int n, int m, int k);
void knn_batch_mlogk_cuda(int n, int m, int k, const float *xyz, const float *query_xyz, const int *batch_idxs, const int *query_batch_offsets, int *idx, cudaStream_t stream);

void knn_batch_mlogk_half(at::Tensor xyz_tensor, at::Tensor query_xyz_tensor, at::Tensor batch_idxs_tensor, at::Tensor query_batch_offsets_tensor, at::Tensor idx_tensor, int n, int m, int k);
void knn_batch_mlogk_cuda_half(int n, int m, int k, const at::BFloat16 *xyz, const at::BFloat16 *query_xyz, const int *batch_idxs, const int *query_batch_offsets, int *idx, cudaStream_t stream);

void knn_batch_mlogk_half_fp16(at::Tensor xyz_tensor, at::Tensor query_xyz_tensor, at::Tensor batch_idxs_tensor, at::Tensor query_batch_offsets_tensor, at::Tensor idx_tensor, int n, int m, int k);
void knn_batch_mlogk_cuda_half_fp16(int n, int m, int k, const at::Half *xyz, const at::Half *query_xyz, const int *batch_idxs, const int *query_batch_offsets, int *idx, cudaStream_t stream);

#endif
