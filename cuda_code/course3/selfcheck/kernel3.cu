#include <chrono>   // for timing
#include <cmath>    // for INFINITY
#include <cstdlib>  // for malloc/free
#include <iostream>

// CPU implementation
void softmax_forward_cpu(float *out, const float *inp, int N, int C) {
  for (int i = 0; i < N; i++) {
    const float *inp_row = inp + i * C;
    float *out_row = out + i * C;

    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
      if (inp_row[j] > maxval) {
        maxval = inp_row[j];
      }
    }
    float sum = 0.f;
    for (int j = 0; j < C; j++) {
      out_row[j] = expf(inp_row[j] - maxval);
      sum += out_row[j];
    }
    float norm = 1.f / sum;
    for (int j = 0; j < C; j++) {
      out_row[j] *= norm;
    }
  }
}

// Function to compare results
bool compare_results(const float *cpu, const float *gpu, int N, int C,
                     float epsilon = 1e-3f) {
  for (int i = 0; i < N * C; ++i) {
    if (fabs(cpu[i] - gpu[i]) > epsilon) {
      std::cout << "Difference at index " << i << ": CPU=" << cpu[i]
                << ", GPU=" << gpu[i] << ", diff=" << fabs(cpu[i] - gpu[i])
                << std::endl;
      return false;
    }
  }
  return true;
}

__device__ float warpReduceMax(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

__device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

__global__ void softmax_forward_kernel3(float *out, const float *inp, int N,
                                        int C) {
  int idx = blockIdx.x;
  int tid = threadIdx.x;
  const float *x = inp + idx * C;

  float maxval = -INFINITY;
  for (int i = tid; i < C; i += blockDim.x) {
    maxval = fmaxf(maxval, x[i]);
  }
  maxval = warpReduceMax(maxval);

  float offset = __shfl_sync(0xFFFFFFFF, maxval, 0);//把lane0的值即maxval广播给其它线程

  for (int i = tid; i < C; i += blockDim.x) {
    out[idx * C + i] = expf(x[i] - offset);
  }

  x = out + idx * C;
  float sumval = 0.0f;
  for (int i = tid; i < C; i += blockDim.x) {
    sumval += x[i];
  }
  sumval = warpReduceSum(sumval);

  float sum = __shfl_sync(0xFFFFFFFF, sumval, 0);

  for (int i = tid; i < C; i += blockDim.x) {
    out[idx * C + i] = x[i] / sum;
  }
}

int main() {
  // Example: batch size N=32, classes C=4096
  int N = 32;
  int C = 4096;

  size_t num_elements = N * C;
  float *inp = (float *)malloc(num_elements * sizeof(float));
  float *out_cpu = (float *)malloc(num_elements * sizeof(float));
  float *out_gpu = (float *)malloc(num_elements * sizeof(float));

  // Initialize input with sample data
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      inp[n * C + c] = float(c);
    }
  }

  // Run CPU version and measure time
  auto start_cpu = std::chrono::high_resolution_clock::now();
  softmax_forward_cpu(out_cpu, inp, N, C);
  auto end_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

  // Run GPU version and measure time using CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float *d_out, *d_inp;
  cudaMalloc((void **)&d_out, N * C * sizeof(float));
  cudaMalloc((void **)&d_inp, N * C * sizeof(float));
  cudaMemcpy(d_inp, inp, N * C * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start);
  // Launch kernel
  int blockSize = 32;
  int numBlocks = N;
  softmax_forward_kernel3<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);
  cudaEventRecord(stop);

  // Wait for the event to complete
  cudaEventSynchronize(stop);

  // Calculate milliseconds
  float gpu_time_ms = 0;
  cudaEventElapsedTime(&gpu_time_ms, start, stop);

  // Copy result back to host
  cudaMemcpy(out_gpu, d_out, N * C * sizeof(float), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_out);
  cudaFree(d_inp);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Compare results
  bool success = compare_results(out_cpu, out_gpu, N, C);
  std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;

  // Print performance comparison
  std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
  std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
  std::cout << "Speedup: " << (cpu_time.count() / (gpu_time_ms)) << "x"
            << std::endl;

  // Cleanup
  free(inp);
  free(out_cpu);
  free(out_gpu);

  return 0;
}