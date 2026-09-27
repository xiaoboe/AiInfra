#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f
// 行优先二维坐标转为一维下标；ld 是一行跨过的元素数量，不是字节数。
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
// 将从指定元素开始的连续 4 个 float 作为一个 float4 读写，地址须满足 16 字节对齐。
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
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
// 计算 C = alpha * A * B + beta * C，原始 A、B、C 均按行优先存储。
// BM、BN：一个 block 的输出块大小；BK：每轮处理的 K 长度。
// TM、TN：一个线程的输出块大小。当前 <128,128,8,8,8> 使用 256 个线程，
// 每线程计算 8x8 个输出，整个 block 覆盖 128x128 个输出。
// 相比 v4：使用 float4 搬运数据，将 A 子块转置存入共享内存，再显式加载
// a_frag、b_frag 供线程内反复使用，最后以 float4 写回输出。
// 此实现没有边界保护：当前参数要求 M、N 为 128 的倍数，K 为 8 的倍数。
// 更换模板参数时还须保证加载分工整除、TM/TN 等向量化维度为 4 的倍数，
// 并满足 float4 地址对齐要求；不能直接用于任意尺寸或任意偏移的指针。
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v6(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // 将线程逻辑排列成 (BM/TM) 行、(BN/TN) 列；当前为 16x16 个线程。
  const int block_row_thread = BN / TN;//行方向上需要多少向量
  const int block_col_thread = BM / TM;//列方向上需要多少向量
  const int thread_num = block_row_thread * block_col_thread;// 当前 block 线程总数；当前为 256 个线程。

  // 当前线程输出小块的局部左上角：tx 为列，ty 为行。
  // 例如线程 17：tx=8、ty=8，负责大块内行 8~15、列 8~15。
  int tx = (threadIdx.x % block_row_thread) * TN;
  int ty = (threadIdx.x / block_row_thread) * TM;

  // As 按转置后的 [K位置][输出行] 存放，形状 BKxBM（8x128）；
  // Bs 按 [K位置][输出列] 存放，形状 BKxBN（8x128）。每个 block 独立拥有。
  // 转置 As 后，固定 K 位置对应的连续输出行数据也连续，便于 float4 读取。
  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];

  // 每线程每轮需要加载多少组 float4。当前 A、B 子块各 1024 个 float，
  // 分给 256 个线程，因此每线程各加载 1 组，即 4 个 A 值和 4 个 B 值。
  const int ldg_a_num = BK * BM / thread_num / 4;
  const int ldg_b_num = BK * BN / thread_num / 4;

  // A 的加载坐标（转置之前）：一行 BK 个元素，需 BK/4 个线程搬运。
  // 当前每行 2 个线程：线程 0 搬 A 的局部行0列0~3，线程1搬行0列4~7。
  // stride 为下一批加载跨过的行数；当前为 128，因此每线程只加载一批。
  int a_tile_row = threadIdx.x / (BK / 4);
  int a_tile_col = threadIdx.x % (BK / 4) * 4;
  int a_tile_stride = BM / ldg_a_num;
  //每一轮一个线程加载lgd_a_num个float4，stride是一组占的行数

  // B 的加载坐标：一行 BN 个元素，需 BN/4=32 个线程搬运。
  // 线程0搬局部行0列0~3，线程32搬行1列0~3；当前 stride=8，只加载一批。
  // 这些是“搬运分工”，与 tx、ty 指定的“输出计算分工”不同。
  int b_tile_row = threadIdx.x / (BN / 4);
  int b_tile_col = threadIdx.x % (BN / 4) * 4;
  int b_tile_stride = BK / ldg_b_num;

  // 每线程独立的输出累加器：当前为 64 个值；跨所有 K 子块持续累加，不清零。
  // 编译器会尽量将这些累加器及下面的临时值放入寄存器，实际分配取决于编译。
  float accum[TM][TN] = {0.};

  // 暂存从 A 连续读入的值，以便拆开后转置写入 As。
  float ldg_a_reg[4 * ldg_a_num] = {0.};

  // 固定一个 K 位置时，当前线程需要的 TM 个 A 值与 TN 个 B 值。
  float a_frag[TM];
  float b_frag[TN];

  // 定位当前 block：A 从对应行块起始，B 从对应列块起始，C 指向输出块左上角。
  // 只移动线程自己的指针，不复制数据；原矩阵的行跨度仍是 K 或 N。
  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];

// 请求编译器展开循环；外层 K 为运行时参数，不保证完全展开。
#pragma unroll
  for (int k = 0; k < K; k += BK) {
    // 从原始 A 的当前 BK 列加载。实际列偏移由下方 A += BK 累计，故下标不再加 k。
#pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
      // 每批占临时数组中的 4 个连续位置。
      int ldg_index = i / a_tile_stride * 4;
      FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
          FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
      // 转置存储：原子块 A[row][col+t] -> As[col+t][row]，t=0,1,2,3。
      // 原来连续的一行 4 个值，拆开写到 As 的 4 行中的同一列。
      As[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
      As[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
      As[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
      As[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
    }
    // B 不转置，保持行优先，连续 4 个值可以直接以 float4 写入 Bs。
#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
      FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)]) =
          FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
    }
    // 等整个 block 加载完成；计算时会读取其他线程搬来的数据。
    __syncthreads();
    // 准备下一轮：A 右移 BK 列，B 下移 BK 行。当前计算使用 As/Bs，不受影响。
    A += BK;
    B += BK * N;
#pragma unroll
    for (int i = 0; i < BK; i++) {
      // i 是当前子块内的 K 位置。As 已转置，因此连续读取的是原 A 的不同行。
      // 当前 TM=8，每线程用两组 float4 取齐自己 8 个输出行需要的 A 值。
#pragma unroll
      for (int m = 0; m < TM; m += 4) {
        FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(i, ty + m, BM)]);
      }
      // 当前 TN=8，同样用两组 float4 取齐自己 8 个输出列需要的 B 值。
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(i, tx + n, BN)]);
      }
      // 外积更新：TM 个 A 值与 TN 个 B 值两两相乘，更新 TMxTN 个累加器。
      // 固定 i：a_frag[m] 复用 TN 次，b_frag[n] 复用 TM 次。
      // 当前 16 个输入值支撑 64 次乘加；i 遍历 BK 后得到当前子块的贡献。
      // m、n 是输出小块内的行列偏移，i 不是输出位置。
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          accum[m][n] += a_frag[m] * b_frag[n];
        }
        //对于这个外积的理解：
        //每次外积计算的是一个 K 位置的贡献
        // a_frag[m] 是当前线程负责的输出小块的第 m 行对应的 A 值，
        // b_frag[n] 是当前线程负责的输出小块的第 n 列对应的 B 值。
        //通过累加这些贡献，最终得到完整的点积结果。
      }
    }
    // 等所有线程读完当前共享内存，再允许下一轮加载覆盖 As/Bs。
    __syncthreads();
  }
  // 所有 K 子块完成后，accum 才是完整点积。每线程写自己的 TMxTN 个输出。
  // 每次读取同一行旧 C 的 4 个值，计算 alpha*累加值+beta*旧值，再向量化写回。
  // C 虽然指向子块起点，但它仍属于原矩阵，因此行跨度使用 N，而不是 BN。
#pragma unroll
  for (int m = 0; m < TM; m++) {
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
      ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
      ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
      ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
      ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
      FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
    }
  }
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
std::vector<int> generateSizes() { return {4096}; }
int main() {
  int device_id = 0;
  checkCudaError(cudaSetDevice(device_id), "cudaSetDevice failed");
  std::vector<int> sizes = generateSizes();
  // 打开CSV文件
  std::ofstream csv_file("sgemm_benchmark_v4.csv");
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

      // mysgemm_v1
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      dim3 blockDim(256);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

      for (int i = 0; i < warpup_time; ++i) {
        mysgemm_v6<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }

      cudaDeviceSynchronize();
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");

      for (int i = 0; i < repeat_time; ++i) {
        mysgemm_v6<128, 128, 8, 8, 8>
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
