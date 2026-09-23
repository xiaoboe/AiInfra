#include <chrono>   // for timing
#include <cmath>    // for INFINITY
#include <cstdlib>  // for malloc/free
#include <iostream>

// CPU implementation
void softmax_forward_cpu(float *out,const float *inp,int N,int C) {
  for (int i = 0;i<N;i++){
    const float *inp_row =  inp + i * C;
    float *out_row =  out + i*C;
    float maxval = -INFINITY;
    //更新最大值
    for (int j = 0;j<C;j++){
      if(inp_row[j] > maxval)
        maxval = inp_row[j];
    }
    //计算分母总和
    float sum = 0.f;
    for(int j = 0;j<C;j++){
      out_row[j] = expf(inp_row[j] - maxval);
      sum += out_row[j];
    }
    float norm = 1.f/sum;
    for (int j = 0;j<C;j++){
      out_row[j] *= norm;
    } 
  }
}


int main() {
  // Example: batch size N=32, classes C=4096
  int N = 32;
  int C = 4096;

  size_t num_elements = N * C;
  float *inp = (float *)malloc(num_elements * sizeof(int));
  float *out_cpu = (float *)malloc(num_elements * sizeof(int));

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

  // Print performance comparison
  std::cout <<"cpu_result:"<< out_cpu <<"\n"<< "CPU time: " << cpu_time.count() << " ms" << std::endl;

  // Cleanup
  free(inp);
  free(out_cpu);

  return 0;
}
