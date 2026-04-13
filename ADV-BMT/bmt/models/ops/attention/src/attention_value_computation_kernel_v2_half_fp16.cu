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
__global__ void attention_value_computation_forward_v2_fp16(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const at::Half *attn_weight, const at::Half* value_features,
    at::Half *output) {
    // dim3 blocks(total_query_num, nhead); dim3 threads(hdim);
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params output: [total_query_num, nhead, hdim]

    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int hdim_idx = threadIdx.x;
    if (query_idx >= total_query_num ||
        head_idx >= nhead ||
        hdim_idx >= hdim) return;

    // get key_start_idx.
    int batch_idx = index_pair_batch[query_idx];
    int key_start_idx = 0;
    for (int i = 0; i < batch_idx; i++){
        key_start_idx += key_batch_cnt[i];
    }
    int cur_key_idx;

    // get shared variables.
    __shared__ at::Half shared_attn_weight[d];  // d == local_size
    __shared__ int shared_value_indices[d];
    for (int i = hdim_idx; i < local_size; i += blockDim.x){
        shared_attn_weight[i] = attn_weight[
            query_idx * local_size * nhead + i * nhead + head_idx];

        cur_key_idx = index_pair[query_idx * local_size + i];
        if (cur_key_idx == -1){
            shared_value_indices[i] = -1;
            continue;
        }
        cur_key_idx += key_start_idx;
        shared_value_indices[i] = cur_key_idx;
    }
    __syncthreads();

    output += query_idx * nhead * hdim + head_idx * hdim + hdim_idx;

    at::Half attn_result = __int2half_rn(0);
    for (int i = 0; i < local_size; i++){
        if (shared_value_indices[i] == -1) continue;
        attn_result = __hadd(attn_result, __hmul(shared_attn_weight[i], value_features[
            shared_value_indices[i] * nhead * hdim + head_idx * hdim + hdim_idx]));
    }
    output[0] = attn_result;
}


void attention_value_computation_launcher_v2_fp16(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const at::Half *attn_weight, const at::Half* value_features,
    at::Half *output){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params output: [total_query_num, nhead, hdim]
    dim3 blocks(total_query_num, nhead);
    dim3 threads(hdim);
    if (local_size > 512){
        throw "local_size should be <= 512.";
    }

    switch (local_size){
        case 16:
            attention_value_computation_forward_v2_fp16<16><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                output);
            break;
        case 32:
            attention_value_computation_forward_v2_fp16<32><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                output);
            break;
        case 64:
            attention_value_computation_forward_v2_fp16<64><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                output);
            break;
        case 128:
            attention_value_computation_forward_v2_fp16<128><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                output);
            break;
        case 320:
            attention_value_computation_forward_v2_fp16<320><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                output);
            break;
        case 384:
            attention_value_computation_forward_v2_fp16<384><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                output);
            break;
        default:
            attention_value_computation_forward_v2_fp16<512><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                output);
            break;
    }
}


template <unsigned int d> // d == local_size
__global__ void attention_value_computation_backward_v2_fp16(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const at::Half *attn_weight, const at::Half* value_features,
    at::Half *grad_out, at::Half * grad_attn_weight, at::Half * grad_value_features) {
    // dim3 blocks(total_query_num, nhead); dim3 threads(hdim);
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params grad_out: [total_query_num, nhead, hdim]
    // params grad_attn_weight: [total_query_num, local_size, nhead]
    // params grad_value_features: [total_key_num, nhead, hdim]
    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int hdim_idx = threadIdx.x;
    if (query_idx >= total_query_num ||
        head_idx >= nhead ||
        hdim_idx >= hdim) return;

    // get key_start_idx.
    int batch_idx = index_pair_batch[query_idx];
    int key_start_idx = 0;
    for (int i = 0; i < batch_idx; i++){
        key_start_idx += key_batch_cnt[i];
    }
    int cur_key_idx;

    // get shared variables.
    __shared__ at::Half shared_attn_weight[d], shared_grad_attn_weight[d];  // d == local_size
    __shared__ int shared_value_indices[d];
    for (int i = hdim_idx; i < local_size; i += blockDim.x){
        shared_attn_weight[i] = attn_weight[
            query_idx * local_size * nhead + i * nhead + head_idx];
        shared_grad_attn_weight[i] = __int2half_rn(0);

        cur_key_idx = index_pair[query_idx * local_size + i];
        if (cur_key_idx == -1){
            shared_value_indices[i] = -1;
            continue;
        }
        cur_key_idx += key_start_idx;
        shared_value_indices[i] = cur_key_idx;
    }
    __syncthreads();

    at::Half gradient = grad_out[query_idx * nhead * hdim + head_idx * hdim + hdim_idx];
    for (int i = 0; i < local_size; i++){
        if (shared_value_indices[i] == -1) continue;

//        atomicAdd(
//            shared_grad_attn_weight + i,
//            __hmul(gradient, value_features[shared_value_indices[i] * nhead * hdim + head_idx * hdim + hdim_idx])
//        );
//        atomicAdd(
//            grad_value_features + shared_value_indices[i] * nhead * hdim + head_idx * hdim + hdim_idx,
//            __hmul(gradient, shared_attn_weight[i])
//        );

        // PZH NOTE: AtomicAdd is extremely slow for FP16 or BF16. We can accelerate code by:
        // PZH NOTE: According to https://github.com/BBuf/how-to-optim-algorithm-in-cuda
        at::native::fastAtomicAdd(
            shared_grad_attn_weight + i,
            0, 0,
            gradient * value_features[shared_value_indices[i] * nhead * hdim + head_idx * hdim + hdim_idx],
            true
        );
        at::native::fastAtomicAdd(
            grad_value_features + shared_value_indices[i] * nhead * hdim + head_idx * hdim + hdim_idx,
            0, 0,
            gradient * shared_attn_weight[i],
            true
        );

//        fastAtomicAdd(gradInput.data(), index, gradInput_numel, val, true);
    }
    __syncthreads();

    for (int i = hdim_idx; i < local_size; i+=blockDim.x){
        grad_attn_weight[query_idx * local_size * nhead + i * nhead + head_idx] = shared_grad_attn_weight[i];
    }
}


void attention_value_computation_grad_launcher_v2_fp16(
    int b, int total_query_num, int local_size,
    int total_key_num, int nhead, int hdim,
    const int *query_batch_cnt, const int *key_batch_cnt, const int* index_pair_batch,
    const int *index_pair,
    const at::Half *attn_weight, const at::Half* value_features,
    at::Half *grad_out, at::Half* grad_attn_weight, at::Half* grad_value_features){
    // params query_batch_cnt: [b]
    // params key_batch_cnt: [b]
    // params index_pair_batch: [total_query_num]
    // params index_pair: [total_query_num, local_size]
    // params attn_weight: [total_query_num, local_size, nhead]
    // params value_features: [total_key_num, nhead, hdim]
    // params grad_out: [total_query_num, nhead, hdim]
    // params grad_attn_weight: [total_query_num, local_size, nhead]
    // params grad_value_features: [total_key_num, nhead, hdim]
    dim3 blocks(total_query_num, nhead);
    dim3 threads(hdim);
    if (local_size > 512){
        throw "local_size should be <= 512.";
    }

    switch(local_size){
        case 16:
            attention_value_computation_backward_v2_fp16<16><<<blocks, threads>>>(
                 b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                grad_out, grad_attn_weight, grad_value_features);
            break;
        case 32:
            attention_value_computation_backward_v2_fp16<32><<<blocks, threads>>>(
                 b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                grad_out, grad_attn_weight, grad_value_features);
            break;
        case 64:
            attention_value_computation_backward_v2_fp16<64><<<blocks, threads>>>(
                 b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                grad_out, grad_attn_weight, grad_value_features);
            break;
        case 128:
            attention_value_computation_backward_v2_fp16<128><<<blocks, threads>>>(
                 b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                grad_out, grad_attn_weight, grad_value_features);
            break;
        case 320:
            attention_value_computation_backward_v2_fp16<320><<<blocks, threads>>>(
                 b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                grad_out, grad_attn_weight, grad_value_features);
            break;
        case 384:
            attention_value_computation_backward_v2_fp16<384><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                grad_out, grad_attn_weight, grad_value_features);
            break;
        default:
           attention_value_computation_backward_v2_fp16<512><<<blocks, threads>>>(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                grad_out, grad_attn_weight, grad_value_features);
           break;
    }
}
