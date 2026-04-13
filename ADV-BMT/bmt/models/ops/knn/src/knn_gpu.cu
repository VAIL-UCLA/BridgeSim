// Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
// Published at NeurIPS 2022
// Written by Li Jiang, Shaoshuai Shi 
// All Rights Reserved


#include "knn_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

__global__ void knn_batch_cuda_(int n, int m, int k, const float *__restrict__ xyz, const float *__restrict__ query_xyz, const int *__restrict__ batch_idxs, const int *__restrict__ query_batch_offsets, int *__restrict__ idx) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= n) return;

    xyz += pt_idx * 3;
    idx += pt_idx * k;

    float ox = xyz[0];
    float oy = xyz[1];
    float oz = xyz[2];

    float best[100];
    int besti[100];
    for(int i = 0; i < k; i++){
        best[i] = 1e20;
        besti[i] = -1;
    }

    int batch_idx = batch_idxs[pt_idx];
    int start = query_batch_offsets[batch_idx];
    int end = query_batch_offsets[batch_idx + 1];

    for (int i = start; i < end; ++i) {
        float x = query_xyz[i * 3 + 0];
        float y = query_xyz[i * 3 + 1];
        float z = query_xyz[i * 3 + 2];
        float d2 = (ox - x) * (ox - x) + (oy - y) * (oy - y) + (oz - z) * (oz - z);
        for(int p = 0; p < k; p++){
            if(d2 < best[p]){
                for(int q = k - 1; q > p; q--){
                    best[q] = best[q - 1];
                    besti[q] = besti[q - 1];
                }
                best[p] = d2;
                besti[p] = i - start;
                break;
            }
        }
    }

    for(int i = 0; i < k; i++){
        idx[i] = besti[i];
    }
}


__global__ void knn_batch_mlogk_cuda_(int n, int m, int k, const float *__restrict__ xyz, const float *__restrict__ query_xyz, const int *__restrict__ batch_idxs, const int *__restrict__ query_batch_offsets, int *__restrict__ idx) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= n) return;

    xyz += pt_idx * 3;
    idx += pt_idx * k;

    float ox = xyz[0];
    float oy = xyz[1];
    float oz = xyz[2];

    float best[150];
    int besti[150];

    int heap_len = 0;

    for(int i = 0; i <= k; i++){
        best[i] = std::numeric_limits<float>::infinity();
        besti[i] = -1;
    }

    int batch_idx = batch_idxs[pt_idx];
    int start = query_batch_offsets[batch_idx];
    int end = query_batch_offsets[batch_idx + 1];
    int temp_i;
    float temp_f;

    for (int i = start; i < end; ++i) {
        float x = query_xyz[i * 3 + 0];
        float y = query_xyz[i * 3 + 1];
        float z = query_xyz[i * 3 + 2];
        float d2 = (ox - x) * (ox - x) + (oy - y) * (oy - y) + (oz - z) * (oz - z);

        if (heap_len < k){
            heap_len++;
            best[heap_len] = d2;
            besti[heap_len] = i - start;
            int cur_idx = heap_len, fa_idx = cur_idx >> 1;

            while (fa_idx > 0){
                if (best[cur_idx] < best[fa_idx]) break;

                temp_i = besti[cur_idx]; besti[cur_idx] = besti[fa_idx]; besti[fa_idx] = temp_i; 
                temp_f = best[cur_idx]; best[cur_idx] = best[fa_idx]; best[fa_idx] = temp_f;
                cur_idx = fa_idx;
                fa_idx = cur_idx >> 1;
            }
        }
        else{
            if (d2 > best[1]) continue;
            best[1] = d2; besti[1] = i - start;

            int cur_idx = 1, son_idx;
            while (cur_idx <= k){
                son_idx = cur_idx << 1;
                if (son_idx > k) break;
                if (son_idx + 1 <= k && best[son_idx] < best[son_idx + 1]){
                    son_idx++;
                }

                if (son_idx <= k && best[cur_idx] < best[son_idx]){
                    temp_i = besti[cur_idx]; besti[cur_idx] = besti[son_idx]; besti[son_idx] = temp_i; 
                    temp_f = best[cur_idx]; best[cur_idx] = best[son_idx]; best[son_idx] = temp_f;
                }
                else break;
                cur_idx = son_idx;
            }
        }
    }
    
    for(int i = 1; i <= k; i++){
        idx[i - 1] = besti[i];
    }
    // delete [] best;
    // delete [] besti;
}



__global__ void knn_batch_mlogk_cuda_half_(int n, int m, int k, const at::BFloat16 *__restrict__ xyz, const at::BFloat16 *__restrict__ query_xyz, const int *__restrict__ batch_idxs, const int *__restrict__ query_batch_offsets, int *__restrict__ idx) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= n) return;

    xyz += pt_idx * 3;
    idx += pt_idx * k;

    at::BFloat16 ox = xyz[0];
    at::BFloat16 oy = xyz[1];
    at::BFloat16 oz = xyz[2];

    at::BFloat16 best[150];
    int besti[150];

    int heap_len = 0;

    for(int i = 0; i <= k; i++){
        best[i] = __float2bfloat16(std::numeric_limits<float>::infinity());
        besti[i] = -1;
    }

    int batch_idx = batch_idxs[pt_idx];
    int start = query_batch_offsets[batch_idx];
    int end = query_batch_offsets[batch_idx + 1];
    int temp_i;
    at::BFloat16 temp_f;

    for (int i = start; i < end; ++i) {
        at::BFloat16 x = query_xyz[i * 3 + 0];
        at::BFloat16 y = query_xyz[i * 3 + 1];
        at::BFloat16 z = query_xyz[i * 3 + 2];
        at::BFloat16 d2 = (ox - x) * (ox - x) + (oy - y) * (oy - y) + (oz - z) * (oz - z);

        if (heap_len < k){
            heap_len++;
            best[heap_len] = d2;
            besti[heap_len] = i - start;
            int cur_idx = heap_len, fa_idx = cur_idx >> 1;

            while (fa_idx > 0){
                if (best[cur_idx] < best[fa_idx]) break;

                temp_i = besti[cur_idx]; besti[cur_idx] = besti[fa_idx]; besti[fa_idx] = temp_i;
                temp_f = best[cur_idx]; best[cur_idx] = best[fa_idx]; best[fa_idx] = temp_f;
                cur_idx = fa_idx;
                fa_idx = cur_idx >> 1;
            }
        }
        else{
            if (d2 > best[1]) continue;
            best[1] = d2; besti[1] = i - start;

            int cur_idx = 1, son_idx;
            while (cur_idx <= k){
                son_idx = cur_idx << 1;
                if (son_idx > k) break;
                if (son_idx + 1 <= k && best[son_idx] < best[son_idx + 1]){
                    son_idx++;
                }

                if (son_idx <= k && best[cur_idx] < best[son_idx]){
                    temp_i = besti[cur_idx]; besti[cur_idx] = besti[son_idx]; besti[son_idx] = temp_i; 
                    temp_f = best[cur_idx]; best[cur_idx] = best[son_idx]; best[son_idx] = temp_f;
                }
                else break;
                cur_idx = son_idx;
            }
        }
    }
    
    for(int i = 1; i <= k; i++){
        idx[i - 1] = besti[i];
    }
    // delete [] best;
    // delete [] besti;
}




__global__ void knn_batch_mlogk_cuda_half_fp16_(int n, int m, int k, const at::Half *__restrict__ xyz, const at::Half *__restrict__ query_xyz, const int *__restrict__ batch_idxs, const int *__restrict__ query_batch_offsets, int *__restrict__ idx) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= n) return;

    xyz += pt_idx * 3;
    idx += pt_idx * k;

    at::Half ox = xyz[0];
    at::Half oy = xyz[1];
    at::Half oz = xyz[2];

    at::Half best[150];
    int besti[150];

    int heap_len = 0;

    for(int i = 0; i <= k; i++){
        best[i] = __float2half(std::numeric_limits<float>::infinity());
        besti[i] = -1;
    }

    int batch_idx = batch_idxs[pt_idx];
    int start = query_batch_offsets[batch_idx];
    int end = query_batch_offsets[batch_idx + 1];
    int temp_i;
    at::Half temp_f;

    for (int i = start; i < end; ++i) {
        at::Half x = query_xyz[i * 3 + 0];
        at::Half y = query_xyz[i * 3 + 1];
        at::Half z = query_xyz[i * 3 + 2];
        at::Half d2 = (ox - x) * (ox - x) + (oy - y) * (oy - y) + (oz - z) * (oz - z);

        if (heap_len < k){
            heap_len++;
            best[heap_len] = d2;
            besti[heap_len] = i - start;
            int cur_idx = heap_len, fa_idx = cur_idx >> 1;

            while (fa_idx > 0){
                if (best[cur_idx] < best[fa_idx]) break;

                temp_i = besti[cur_idx]; besti[cur_idx] = besti[fa_idx]; besti[fa_idx] = temp_i;
                temp_f = best[cur_idx]; best[cur_idx] = best[fa_idx]; best[fa_idx] = temp_f;
                cur_idx = fa_idx;
                fa_idx = cur_idx >> 1;
            }
        }
        else{
            if (d2 > best[1]) continue;
            best[1] = d2; besti[1] = i - start;

            int cur_idx = 1, son_idx;
            while (cur_idx <= k){
                son_idx = cur_idx << 1;
                if (son_idx > k) break;
                if (son_idx + 1 <= k && best[son_idx] < best[son_idx + 1]){
                    son_idx++;
                }

                if (son_idx <= k && best[cur_idx] < best[son_idx]){
                    temp_i = besti[cur_idx]; besti[cur_idx] = besti[son_idx]; besti[son_idx] = temp_i; 
                    temp_f = best[cur_idx]; best[cur_idx] = best[son_idx]; best[son_idx] = temp_f;
                }
                else break;
                cur_idx = son_idx;
            }
        }
    }
    
    for(int i = 1; i <= k; i++){
        idx[i - 1] = besti[i];
    }
    // delete [] best;
    // delete [] besti;
}




void knn_batch_cuda(int n, int m, int k, const float *xyz, const float *query_xyz, const int *batch_idxs, const int *query_batch_offsets, int *idx, cudaStream_t stream) {
    // param xyz: (n, 3), float
    // param query_xyz: (m, 3), float
    // param batch_idxs: (n), int
    // param query_batch_offsets: (B + 1), int, offsets[-1] = m
    // param idx: (n, k), int

    cudaError_t err;

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    knn_batch_cuda_<<<blocks, threads, 0, stream>>>(n, m, k, xyz, query_xyz, batch_idxs, query_batch_offsets, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


void knn_batch_mlogk_cuda(int n, int m, int k, const float *xyz, const float *query_xyz, const int *batch_idxs, const int *query_batch_offsets, int *idx, cudaStream_t stream) {
    // param xyz: (n, 3), float
    // param query_xyz: (m, 3), float
    // param batch_idxs: (n), int
    // param query_batch_offsets: (B + 1), int, offsets[-1] = m
    // param idx: (n, k), int

    cudaError_t err;

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    knn_batch_mlogk_cuda_<<<blocks, threads, 0, stream>>>(n, m, k, xyz, query_xyz, batch_idxs, query_batch_offsets, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


void knn_batch_mlogk_cuda_half(int n, int m, int k, const at::BFloat16 *xyz, const at::BFloat16 *query_xyz, const int *batch_idxs, const int *query_batch_offsets, int *idx, cudaStream_t stream) {
    // param xyz: (n, 3), at::BFloat16
    // param query_xyz: (m, 3), at::BFloat16
    // param batch_idxs: (n), int
    // param query_batch_offsets: (B + 1), int, offsets[-1] = m
    // param idx: (n, k), int

    cudaError_t err;

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    knn_batch_mlogk_cuda_half_<<<blocks, threads, 0, stream>>>(n, m, k, xyz, query_xyz, batch_idxs, query_batch_offsets, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


void knn_batch_mlogk_cuda_half_fp16(int n, int m, int k, const at::Half *xyz, const at::Half *query_xyz, const int *batch_idxs, const int *query_batch_offsets, int *idx, cudaStream_t stream) {
    // param xyz: (n, 3), at::Half
    // param query_xyz: (m, 3), at::Half
    // param batch_idxs: (n), int
    // param query_batch_offsets: (B + 1), int, offsets[-1] = m
    // param idx: (n, k), int

    cudaError_t err;

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    knn_batch_mlogk_cuda_half_fp16_<<<blocks, threads, 0, stream>>>(n, m, k, xyz, query_xyz, batch_idxs, query_batch_offsets, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
