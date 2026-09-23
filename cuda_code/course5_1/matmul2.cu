#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void checkCublasError(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  int bx = blockIdx.x;
  int by = blockIdx.y;//线程块行索引
// BM	128	每个 block 负责的输出行数
// BN	128	每个 block 负责的输出列数
// BK	8	每轮处理的 K 维度长度
// TM	8	每个线程负责的输出行数
// TN	8	每个线程负责的输出列数
  int block_row_thread = BN / TN;//横向需要线程数
  // 比如一个输出块有 128 列，每个线程负责 8 列,因此拿列相除
  int block_col_thread = BM / TM;//纵向需要线程数
  int thread_num = block_row_thread * block_col_thread;//需要总线程数

// 当前线程负责的 8 × 8 输出小块，在当前 block 的输出大块中的左上角坐标。
  int tx = (threadIdx.x % block_row_thread) * TN;
  int ty = (threadIdx.x / block_row_thread) * TM;

  __shared__ float As[BM * BK];//这里存储的是一整块,而不是一整行
  __shared__ float Bs[BK * BN];

  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];
//~~~
  int a_tile_row = threadIdx.x / BK;//第一次加载的行
  int a_tile_col = threadIdx.x % BK;
  int a_tile_stride = thread_num / BK;//下次加载向下跨多少行

  int b_tile_row = threadIdx.x / BN;
  int b_tile_col = threadIdx.x % BN;
  int b_tile_stride = thread_num / BN;

  float tmp[TM][TN] = {0.};
#pragma unroll
  for (int k = 0; k < K; k += BK) {
#pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
      As[(a_tile_row + i) * BK + a_tile_col] =
          A[(a_tile_row + i) * K + a_tile_col];
    }
#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
      Bs[(b_tile_row + i) * BN + b_tile_col] =
          B[(b_tile_row + i) * N + b_tile_col];
    }
//这两段是在把当前这一轮计算需要的 A 子块和 B 子块，从全局内存加载到共享内存 As、Bs
//其中每个线程加载BM*BK/thread_num个元素
    __syncthreads();
    A += BK;//A 向右移动 BK 列：
    B += BK * N;//B 向下移动 BK 行：
#pragma unroll
    for (int i = 0; i < BK; i++) {
#pragma unroll
      for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++)
          tmp[j][l] += As[(ty + j) * BK + i] * Bs[(tx + l) + i * BN];
          //tmp[j][l] = As[ty+j][i] * Bs[i][tx+l]
          //这个i,j是在TM,TN这个小块内的偏移量
          //这里可以从外积的角度考虑
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < TM; j++) {
    for (int l = 0; l < TN; l++)
      C[(ty + j) * N + tx + l] =
          alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
  }
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

std::vector<int> generateSizes() { return {4096}; }
int main() {
  int device_id = 7;
  checkCudaError(cudaSetDevice(device_id), "cudaSetDevice failed");
  std::vector<int> sizes = generateSizes();

  // 打开CSV文件
  std::ofstream csv_file("sgemm_benchmark_v3.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

  for (int N : sizes) {
    std::cout << "Testing size: " << N << std::endl;

    size_t size = N * N * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C_cublas = (float *)malloc(size);
    float *C_v1 = (float *)malloc(size);

    float *d_A, *d_B, *d_C_v1;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");

    bool out_of_memory = false;

    try {
      // 初始化矩阵 A 和 B
      for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
      }

      // 拷贝到设备
      checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy A to device failed");
      checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy B to device failed");

      cublasHandle_t handle;
      checkCublasError(cublasCreate(&handle), "cublasCreate failed");

      float alpha = 1.0f;
      float beta = 0.0f;

      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
      checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

      // warmup
      int warpup_time = 10;  // 热身次数
      for (int i = 0; i < warpup_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 5;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }

      checkCudaError(cudaEventRecord(stop),
                     "cudaEventRecord(stop cublas) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize cublas failed");

      float cublas_time = 0;
      checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                     "cudaEventElapsedTime cublas failed");

      // 拷贝 cuBLAS 结果
      checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");

      // mysgemm_v4
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      dim3 blockDim(256);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

      for (int i = 0; i < warpup_time; ++i) {
        mysgemm_v4<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }

      cudaDeviceSynchronize();
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");

      for (int i = 0; i < repeat_time; ++i) {
        mysgemm_v4<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v1 failed");
      float v1_time = 0;
      checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                     "cudaEventElapsedTime v1 failed");

      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v1 failed");
      // 结果比较
      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i) {
        if (fabsf(C_cublas[i] - C_v1[i]) > TOL) {
          error_count++;
        }
      }

      float cublas_gflops =
          repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // GFlops
      float v1_gflops =
          repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // GFlops
      // 写入CSV
      csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
               << (error_count == 0 ? "1" : "0") << std::endl;

      // 释放资源
      cublasDestroy(handle);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C_v1);

      free(A);
      free(B);
      free(C_cublas);
      free(C_v1);

    } catch (...) {
      std::cerr << "Out of memory or error during testing size: " << N
                << std::endl;
      out_of_memory = true;
    }

    if (!out_of_memory) {
      std::cout << "Finished size: " << N << std::endl;
    } else {
      csv_file << N << ",OOM,OOM,0" << std::endl;
    }
  }

  csv_file.close();

  std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
            << std::endl;
  return 0;
}
