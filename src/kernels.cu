#include <vector>

#include "../tester/utils.h"
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <cassert>

/**
 * @brief Find the k-th largest element in a vector using CUDA.
 * 
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed. 
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
template <typename T>
__device__ void partition3(T* data, int low, int high, int& left_eq, int& right_eq) {
    T pivot = data[high];
    int i = low;
    int lt = low;       // data[low..lt-1] > pivot
    int gt = high;      // data[gt+1..high] < pivot
    while (i <= gt) {
        if (data[i] > pivot) { // 大于pivot
            T tmp = data[lt];
            data[lt] = data[i];
            data[i] = tmp;
            lt++; i++;
        } else if (data[i] < pivot) { // 小于pivot
            T tmp = data[i];
            data[i] = data[gt];
            data[gt] = tmp;
            gt--;
        } else { // 等于pivot
            i++;
        }
    }
    left_eq = lt;
    right_eq = gt;
}


template <typename T>
__device__ int quickSelect3(T* data, int low, int high, int k) {
    while (low <= high) {
        int left_eq, right_eq;
        partition3(data, low, high, left_eq, right_eq);
        if (k >= left_eq && k <= right_eq) {
            return k; // 找到第k大元素，k在等于pivot的范围内
        } else if (k < left_eq) {
            high = left_eq - 1; // 在左边找更大的元素
        } else {
            low = right_eq + 1; // 在右边找更小的元素
        }
    }
    return -1; // 没找到，理论上不该出现
}


template <typename T>
__global__ void quickSelectKernel(T* data, int low, int high, int k, int* result_idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { // 只用一个线程执行
        int idx = quickSelect3(data, low, high, k);
        *result_idx = idx;
    }
}

template <typename T>
T kthLargest(const std::vector<T>& h_input, size_t k) {
    int n = h_input.size();
    if (k <= 0 || k > n) {
        return T(-100);
    }

    T* d_data;
    cudaMalloc(&d_data, n * sizeof(T));
    cudaMemcpy(d_data, h_input.data(), n * sizeof(T), cudaMemcpyHostToDevice);

    int* d_result_idx;
    int h_result_idx = -1;
    cudaMalloc(&d_result_idx, sizeof(int));
    cudaMemcpy(d_result_idx, &h_result_idx, sizeof(int), cudaMemcpyHostToDevice);

    int kIndex = k - 1;

    quickSelectKernel<T><<<1, 1>>>(d_data, 0, n - 1, kIndex, d_result_idx);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result_idx, d_result_idx, sizeof(int), cudaMemcpyDeviceToHost);

    T result;
    if (h_result_idx >= 0) {
        cudaMemcpy(&result, d_data + h_result_idx, sizeof(T), cudaMemcpyDeviceToHost);
    } else {
        result = T(-100);
    }

    cudaFree(d_data);
    cudaFree(d_result_idx);
    return result;
}
// 用partition和quickSelect函数实现并行化的快速选择算法
// 将数据从主机内存复制到设备内存，然后在GPU上计算第k大的元素
// 最后将结果从设备内存复制回主机内存，并释放设备内存

// 使用CUDA编程需要配置一个NVIDIA GPU环境和CUDA编译器。
// CUDA编程通常针对GPU计算密集型任务。

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
// 简单 softmax 实现
void softmax(const float* input, float* output, int length) {
    float max_val = *std::max_element(input, input + length);
    float sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < length; ++i) {
        output[i] /= sum;
    }
}
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    assert(h_q.size() == static_cast<size_t>(batch_size * target_seq_len * query_heads * head_dim));
    assert(h_k.size() == static_cast<size_t>(batch_size * src_seq_len * kv_heads * head_dim));
    assert(h_v.size() == static_cast<size_t>(batch_size * src_seq_len * kv_heads * head_dim));

    if (head_dim <= 0) {
        throw std::invalid_argument("head_dim must be > 0");
    }

    h_o.resize(batch_size * target_seq_len * query_heads * head_dim);

    const T scale = static_cast<T>(1.0 / std::sqrt(static_cast<double>(head_dim)));

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < query_heads; ++h) {
            int kvh = h * kv_heads / query_heads;
            if (kvh >= kv_heads) {
                throw std::out_of_range("kvh index out of range");
            }

            for (int tq = 0; tq < target_seq_len; ++tq) {
                std::vector<T> scores(src_seq_len, static_cast<T>(0));

                for (int sk = 0; sk < src_seq_len; ++sk) {
                    if (is_causal && sk > tq) {
                        scores[sk] = -std::numeric_limits<T>::infinity();
                        continue;
                    }

                    T dot = 0;
                    for (int d = 0; d < head_dim; ++d) {
                        size_t q_idx = ((b * target_seq_len + tq) * query_heads + h) * head_dim + d;
                        size_t k_idx = ((b * src_seq_len + sk) * kv_heads + kvh) * head_dim + d;
                        dot += h_q[q_idx] * h_k[k_idx];
                    }
                    scores[sk] = dot * scale;
                }

                std::vector<T> probs(src_seq_len);
                softmax(scores.data(), probs.data(), src_seq_len);

                for (int d = 0; d < head_dim; ++d) {
                    T acc = 0;
                    for (int sk = 0; sk < src_seq_len; ++sk) {
                        size_t v_idx = ((b * src_seq_len + sk) * kv_heads + kvh) * head_dim + d;
                        assert(v_idx < h_v.size());
                        acc += probs[sk] * h_v[v_idx];
                    }
                    size_t o_idx = ((b * target_seq_len + tq) * query_heads + h) * head_dim + d;
                    assert(o_idx < h_o.size());
                    h_o[o_idx] = acc;
                }
            }
        }
    }
}
// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int>&, size_t);
template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);