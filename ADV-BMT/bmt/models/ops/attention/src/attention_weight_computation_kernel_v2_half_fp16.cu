/*
Transformer function helper function.
Written by tomztyang,
2021/08/23
*/

#include <math.h>
#include <stdio.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
// #define DEBUG


template <unsigned int d>
__global__ void attention_weight_computation_forward_v2_fp16(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const at::Half *query_features, const at::Half* key_features,
    at::Half *output) {
    // dim3 blocks(total_query_num, nhead); dim3 threads(local_size);
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params query_features: [total_query_num, nhead, hdim]
    // params key_features: [total_key_num, nhead, hdim]
    // params output: [total_query_num, local_size, nhead]

    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int local_key_idx = threadIdx.x;

    int index = query_idx * local_size + local_key_idx;
    if (query_idx >= total_query_num ||
        head_idx >= nhead ||
        local_key_idx >= local_size) return;

    // build shared query features.
    __shared__ at::Half shared_query_features[d];
    for (int i = local_key_idx; i < hdim; i += blockDim.x){
        shared_query_features[i] = query_features[
            query_idx * nhead * hdim + head_idx * hdim + i];
    }
    __syncthreads();

    if (index_pair[index] == -1){
        // Ignore index.
        return;
    }

    // get real key_idx.
    int batch_idx = index_pair_batch[query_idx];
    int key_start_idx = 0;
    for (int i = 0; i < batch_idx; i++){
        key_start_idx += key_batch_cnt[i];
    }
    key_start_idx += index_pair[index];

    // get key features.
    key_features += key_start_idx * nhead * hdim + head_idx * hdim;
    output += index * nhead + head_idx;

    at::Half attn_weight = __int2half_rn(0);
    for (int i = 0; i < hdim; i++){
        attn_weight = __hadd(attn_weight, __hmul(key_features[i], shared_query_features[i]));
    }
    output[0] = attn_weight;
}


void attention_weight_computation_launcher_v2_fp16(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const at::Half *query_features, const at::Half* key_features,
    at::Half *output){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params query_features: [total_query_num, nhead, hdim]
    // params key_features: [total_key_num, nhead, hdim]
    // params output: [total_query_num, local_size, nhead]
    if (hdim > 150){
        throw "hdim should be <= 150.";
    }

    dim3 blocks(total_query_num, nhead);
    dim3 threads(local_size);
    switch(hdim){  // switch hdim for utilizing different shared vectors.
        case 16:
            attention_weight_computation_forward_v2_fp16<16><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                output);
            break;
        case 24:
            attention_weight_computation_forward_v2_fp16<24><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                output);
            break;
        case 32:
            attention_weight_computation_forward_v2_fp16<32><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                output);
            break;
        case 48:
            attention_weight_computation_forward_v2_fp16<48><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                output);
            break;
        case 64:
            attention_weight_computation_forward_v2_fp16<64><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                output);
            break;
        case 128:
            attention_weight_computation_forward_v2_fp16<128><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                output);
            break;
        default:
            attention_weight_computation_forward_v2_fp16<150><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                output);
            break;
    }
}


template <unsigned int d>
__global__ void attention_weight_computation_backward_v2_fp16(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const at::Half *query_features, const at::Half* key_features,
    at::Half *grad_out, at::Half * grad_query_features, at::Half * grad_key_features) {
    // dim3 blocks(total_query_num, nhead); dim3 threads(local_size);
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params query_features: [total_query_num, nhead, hdim]
    // params key_features: [total_key_num, nhead, hdim]
    // params grad_out: [total_query_num, local_size, nhead]
    // params grad_query_features: [total_query_num, nhead, hdim]
    // params grad_key_features: [total_key_num, nhead, hdim]

    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int local_key_idx = threadIdx.x;
    int index = query_idx * local_size + local_key_idx;

    if (query_idx >= total_query_num ||
        head_idx >= nhead ||
        local_key_idx >= local_size) return;

    // build shared query features.
    __shared__ at::Half shared_query_features[d];
    __shared__ at::Half shared_grad_query_features[d];
    for (int i = local_key_idx; i < hdim; i += blockDim.x){
        shared_query_features[i] = query_features[
            query_idx * nhead * hdim + head_idx * hdim + i];
        shared_grad_query_features[i] = __int2half_rn(0);
    }
    __syncthreads();

    if (index_pair[index] != -1){
        int batch_idx = index_pair_batch[query_idx];
        int key_start_idx = 0;
        for (int i = 0; i < batch_idx; i++){
            key_start_idx += key_batch_cnt[i];
        }
        key_start_idx += index_pair[index];

        key_features += key_start_idx * nhead * hdim + head_idx * hdim;
        grad_key_features += key_start_idx * nhead * hdim + head_idx * hdim;

        at::Half gradient = grad_out[index * nhead + head_idx];
        for (int i = 0; i < hdim; i++){
//            atomicAdd(
//                shared_grad_query_features + i,
//                gradient * key_features[i]);
//            atomicAdd(
//                grad_key_features + i,
//                gradient * shared_query_features[i]);
            at::native::fastAtomicAdd(
                shared_grad_query_features + i,
                0, 0,
                gradient * key_features[i], true);
            at::native::fastAtomicAdd(
                grad_key_features + i, 0, 0,
                gradient * shared_query_features[i], true);
        }
    }
    __syncthreads();

    grad_query_features += query_idx * nhead * hdim + head_idx * hdim;
    for (int i = local_key_idx; i < hdim; i += blockDim.x){
        grad_query_features[i] = shared_grad_query_features[i];
    }
}


void attention_weight_computation_grad_launcher_v2_fp16(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const at::Half *query_features, const at::Half* key_features,
    at::Half *grad_out, at::Half* grad_query_features, at::Half* grad_key_features){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params query_features: [total_query_num, nhead, hdim]
    // params key_features: [total_key_num, nhead, hdim]
    // params grad_out: [total_query_num, local_size, nhead]
    // params grad_query_features: [total_query_num, nhead, hdim]
    // params grad_key_features: [total_key_num, nhead, hdim]
    if (hdim > 150){
        throw "hdim should be <= 150.";
    }

    dim3 blocks(total_query_num, nhead);
    dim3 threads(local_size);

    switch(hdim){
        case 16:
            attention_weight_computation_backward_v2_fp16<16><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                grad_out, grad_query_features, grad_key_features);
            break;
        case 24:
            attention_weight_computation_backward_v2_fp16<24><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                grad_out, grad_query_features, grad_key_features);
            break;
        case 32:
            attention_weight_computation_backward_v2_fp16<32><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                grad_out, grad_query_features, grad_key_features);
            break;
        case 48:
            attention_weight_computation_backward_v2_fp16<48><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                grad_out, grad_query_features, grad_key_features);
            break;
        case 64:
            attention_weight_computation_backward_v2_fp16<64><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                grad_out, grad_query_features, grad_key_features);
            break;
        case 128:
            attention_weight_computation_backward_v2_fp16<128><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                grad_out, grad_query_features, grad_key_features);
            break;
        default:
            attention_weight_computation_backward_v2_fp16<150><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                grad_out, grad_query_features, grad_key_features);
            break;
    }
}