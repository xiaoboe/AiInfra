// rmsnorm_cuda_test.cpp
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

void row_rmsnorm_f32_dim_cpu(float* in, float* weight, float* out, int batch,
                             int size, float eps) {
  for (int i = 0; i < batch; ++i) {
    float* in_ptr = in + i * size;
    float* out_ptr = out + i * size;

    float sum = 0.0f;
    for (int j = 0; j < size; ++j) {
      float val = in_ptr[j];
      sum += val * val;
    }
    float rms = 1.0f / std::sqrt(sum / static_cast<float>(size) + eps);

    for (int j = 0; j < size; ++j) {
      float x = in_ptr[j] * weight[j];
      out_ptr[j] = x * rms;
    }
  }
}

// 将 block 内每个线程的局部和相加；只有线程 0 的返回值保证是整个 block 的总和。
// 使用条件：一维 block，线程数是 32 的整数倍（32～1024），所有线程都调用此函数。
// 这里使用完整 warp 掩码，不支持直接用不足 32 个线程的 warp 参与归约。
__inline__ __device__ float block_reduce(float val) {
  const int tid = threadIdx.x;
  const int warpSize = 32;
  int lane = tid % warpSize;     // 当前线程在 warp 内的编号：0～31
  int warp_id = tid / warpSize;  // 当前线程属于 block 中的第几个 warp

  // 第一级：各 warp 独立求和。val 是每个线程自己的寄存器变量。
  // offset 依次为 16、8、4、2、1，从 lane + offset 取得对方的 val 并累加。
  // 完成后，每个 warp 的 lane 0 持有该 warp 内 32 个初始 val 的总和。
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);

  // 每个 warp 只由 lane 0 写入一个局部和；共享内存用于跨 warp 传递结果。
  __shared__ float warpSums[32];  // 最多 1024 个线程，即 32 个 warp
  if (lane == 0) {
    warpSums[warp_id] = val;
  }
  // 等待所有 warp 写完，之后第 0 个 warp 才能读取这些局部和。
  __syncthreads();

  // 第二级：仅第 0 个 warp 将所有 warp 的局部和再归约一次。
  if (warp_id == 0) {
    // 向上取整计算 warp 数；不足 32 个局部和时，其余 lane 填 0。
    // 注意：这里的向上取整不能消除第一级对完整 warp 的要求。
    val = (tid < (blockDim.x + warpSize - 1) / warpSize) ? warpSums[tid] : 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  } else {
    // 其他 warp 不参与第二级归约，返回 0。
    val = 0.0f;
  }
  // 第 0 个 warp 的其他 lane 不保证得到总和，调用者只应从线程 0 取结果。
  // 若整个 block 都需要总和，调用者还要通过共享内存和同步进行广播。
  return val;
}

__global__ void row_rmsnorm_f32_dim_simd(float* in, float* wei, float* out,
                                         int batch, int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= batch) {
    return;
  }

  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(block_in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += block_in[i] * block_in[i];
  }

  __shared__ float shared_val;
  sum = block_reduce(sum);

  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(block_out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) = make_float4(
        scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
        scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    block_out[i] = wei[i] * block_in[i] * scale;
  }
}

__global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out,
                                    int batch, int size, float eps) {
  const int bid = blockIdx.x;
  if (bid >= batch) return;

  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  float sum = 0.0f;

  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float x = block_in[i];
    sum += x * x;
  }
  __shared__ float shared_val;
  sum = block_reduce(sum);

  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    float x = block_in[i] * wei[i];
    block_out[i] = x * scale;
  }
}

float compute_max_error(const std::vector<float>& cpu_out,
                        const std::vector<float>& cuda_out, int n) {
  float max_err = 0.0f;
  for (int i = 0; i < n; ++i) {
    float err = std::abs(cpu_out[i] - cuda_out[i]);
    max_err = std::max(max_err, err);
    if (max_err > 1.f) {
      std::cout << "Error at index " << i << ": CPU = " << cpu_out[i]
                << ", CUDA = " << cuda_out[i] << ", Error = " << err << "\n";
      break;
    }
  }
  return max_err;
}

// ----------------------------
// Main Function
// ----------------------------
int main() {
  const int batch = 16;
  const int size = 1024;
  const float eps = 1e-6f;
  const int total = batch * size;

  // Host memory
  std::vector<float> h_input(total);
  std::vector<float> h_weight(size);
  std::vector<float> h_output_cpu(total);
  std::vector<float> h_output_cuda(total);

  // Random init
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dis(0.0f, 1.0f);

  for (int i = 0; i < total; ++i) {
    h_input[i] = dis(gen);
  }
  for (int i = 0; i < size; ++i) {
    h_weight[i] = dis(gen);
  }

  // CPU version
  auto start = std::chrono::high_resolution_clock::now();
  row_rmsnorm_f32_dim_cpu(h_input.data(), h_weight.data(), h_output_cpu.data(),
                          batch, size, eps);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "CPU RMSNorm took " << duration.count() << " microseconds.\n";

  // CUDA setup
  float *d_input, *d_weight, *d_output;
  cudaMalloc(&d_input, total * sizeof(float));
  cudaMalloc(&d_weight, size * sizeof(float));
  cudaMalloc(&d_output, total * sizeof(float));

  cudaMemcpy(d_input, h_input.data(), total * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);

  // Kernel launch config
  const int block_size = 1024;
  const int grid_size = batch;  // One block per batch row
  dim3 grid(grid_size);
  dim3 block(block_size);

  // CUDA timing with events
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);

  int warpup = 10;
  for (int i = 0; i < warpup; i++) {
    // Warm-up run
    row_rmsnorm_f32_dim<<<grid, block>>>(d_input, d_weight, d_output, batch,
                                         size, eps);
  }
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != 0) {
    printf("cuda error:%d\n", err);
  }
  cudaEventRecord(start_event);
  // row_rmsnorm_f32_dim<<<grid, block>>>(d_input, d_weight, d_output, batch,
  // size, eps);
  int test_iter = 10;
  for (int i = 0; i < test_iter; ++i) {
    row_rmsnorm_f32_dim<<<grid, block>>>(d_input, d_weight, d_output, batch,
                                         size, eps);
  }
  cudaEventRecord(stop_event);

  // Wait and measure
  cudaEventSynchronize(stop_event);
  float cuda_time;
  cudaEventElapsedTime(&cuda_time, start_event, stop_event);  // ms

  // Copy result back
  cudaMemcpy(h_output_cuda.data(), d_output, total * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::cout << "CUDA RMSNorm took " << cuda_time * 1000 / test_iter
            << " microseconds.\n";

  // Compare results
  float max_error = compute_max_error(h_output_cpu, h_output_cuda, total);
  std::cout << "Max absolute error (CPU vs CUDA): " << max_error << "\n";

  // Optional: print first few values
  std::cout << "\nFirst 10 outputs (CPU vs CUDA):\n";
  for (int i = 0; i < 10; ++i) {
    std::cout << "CPU: " << h_output_cpu[i] << " | CUDA: " << h_output_cuda[i]
              << " | Diff: " << std::abs(h_output_cpu[i] - h_output_cuda[i])
              << "\n";
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);

  return 0;
}
