# Course 7：CUDA 矩阵转置优化

配套交互页面：[transpose_visualizer.html](transpose_visualizer.html)。直接用浏览器打开即可，页面不依赖网络或第三方库。可切换算法、线程块尺寸、线程编号和块坐标，逐步观察加载、同步、写出；bank 面板独立使用源码真实的 32 × 16 线程块。

## 1. 文件与执行入口

| 文件 | 内容 |
| --- | --- |
| `naive_transpose.cu` | 4 × 4 输入、2 × 2 线程块，打印输入和转置结果 |
| `transpose_bench.cu` | 四个转置 kernel 及对应 benchmark wrapper |
| `CMakeLists.txt` | 定义 `naive_transpose`、`transpose_bench` 两个构建目标 |

本目录优化的是**矩阵转置**，不涉及矩阵乘法。以下基于当前源码推导，没有实测性能排名。`transpose_bench.cu` 的 `main()` 当前只启用了展开版本，其余三项调用被注释。

## 2. 数据布局与基础版本

输入 A 有 `ny` 行、`nx` 列，输出 B 有 `nx` 行、`ny` 列，均按行优先存储，所有下标从 0 开始：

```cpp
B[x][y] = A[y][x];
out[x * ny + y] = in[y * nx + x];
```

`blockDim.x/y` 是一个线程块沿 x/y 的线程数量；`threadIdx.x/y` 是线程的块内坐标；`blockIdx.x/y` 是块在 grid 中的坐标。x 表示列，y 表示行。

`naiveGmem` 计算全局输入坐标：

```cpp
ix = blockIdx.x * blockDim.x + threadIdx.x;
iy = blockIdx.y * blockDim.y + threadIdx.y;
if (ix < nx && iy < ny)
    out[ix * ny + iy] = in[iy * nx + ix];
```

相邻 x 线程读取相邻 float，但写入地址相差 `ny` 个 float。大矩阵时，这种跨行写入会分散全局内存访问。benchmark 的基础版本采用 32 × 32 线程块，演示文件采用 2 × 2。

## 3. 四个版本的优化路线

| kernel | 线程块 | 共享内存 | 每线程元素数 | 主要变化 |
| --- | --- | --- | --- | --- |
| `naiveGmem` | benchmark 为 32 × 32 | 无 | 1 | 直接读写；读连续、写跨行 |
| `transposeSmem` | 32 × 16 | `float tile[16][32]`，2048 B | 1 | 通过共享内存交换搬运任务，改善全局写入布局 |
| `transposeSmemUnpad` | 32 × 16 | `float tile[16][33]`，2112 B | 1 | 实际有 padding，名称 Unpad 与行为不一致 |
| `transposeSmemUnrollPad` | 32 × 16 | `float tile[16 * 65]`，4160 B | 2 | 每块覆盖 16 × 64 输入区域，横向展开两份并补一列 |

共享内存版本并未减少每个元素的一次全局读、一次全局写；它改变的是访问组织。展开版也没有把总数据量减半，而是在完整块下让一个块承担原来两个块的工作。最终收益取决于实际 GPU、访存布局、资源占用和矩阵大小。

## 4. 共享内存索引为什么正确

记 `tx=threadIdx.x`、`ty=threadIdx.y`、`W=32`、`H=16`。输入块起点：

```cpp
R = blockIdx.y * H;  // 行起点
C = blockIdx.x * W;  // 列起点
```

### 第一步：按输入行加载

```cpp
tile[ty][tx] = A[R + ty][C + tx];
__syncthreads();
```

同步完成后，`tile[r][c]` 保存 `A[R+r][C+c]`。共享内存的物理布局没有自动转置。

### 第二步：为线程重新分配输出位置

```cpp
bidx = ty * W + tx;
irow = bidx / H;
icol = bidx % H;
```

把 `H × W` 布局的线程编号重新解释为 `W × H` 布局的输出位置：编号仍为 0～511，输出每行 H=16 个位置，因此除以 16 得到行，余数得到列。这是任务重新分配，不是简单交换当前线程的 tx、ty。

### 第三步：读取别的线程加载的数据并写出

```cpp
ix = R + icol;   // 输出列
iy = C + irow;   // 输出行
to = iy * ny + ix;
out[to] = tile[icol][irow];
```

代入共享内存的含义：

```text
B[C+irow][R+icol] = A[R+icol][C+irow]
```

等式两边恰好交换行列，这证明元素位置正确。又因为 `bidx=irow*H+icol` 唯一，512 个线程恰好覆盖 32 × 16 个输出位置，没有重复或遗漏。以上针对完整块，边缘块需正确设置加载和写出条件。

### 具体例子

输入 `ny=48, nx=128`；线程块 `(blockIdx.x,blockIdx.y)=(2,1)`；线程 `(tx,ty)=(3,2)`：

```text
R=16，C=64
加载：tile[2][3] = A[18][67]，输入线性地址 18*128+67=2371
bidx=2*32+3=67
irow=67/16=4，icol=67%16=3
写出：tile[3][4] = A[19][68] → B[68][19]
to=68*48+19=3283
```

该线程加载和写出的不是同一个元素；`tile[3][4]` 由线程 `(4,3)` 加载。同步保证消费者读取时，生产者已完成写入。

### 全局写入如何改善

对一个 warp，前 16 个 lane 写输出某行的连续 16 个 float，后 16 个 lane 写下一行的连续 16 个 float。这里不是 32 个 lane 必然写入同一段连续地址；但与基础版本每个 lane 跨一整行的写法相比，访问更集中。实际事务还受行跨度和对齐影响。

## 5. Padding 与 bank 冲突：本代码的特殊点

采用 32 个 bank、每个 float 为 4 B 的地址模型，共享内存元素的 bank 为：

```text
bank = (行号 * 物理行跨度 + 列号) % 32
```

同一个 warp 在一条标量共享内存访问中访问同一 bank 的不同地址，会产生 bank 冲突；同地址广播不应按不同地址冲突统计。硬件背景见 [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#shared-memory-and-memory-banks)。

以下结论是对本目录实际索引的推导。取第一个 warp 的共享内存读取 `tile[icol][irow]`：

| 物理行跨度 | lane 0～15 的 bank | lane 16～31 的 bank | 最大不同地址数 / bank |
| --- | --- | --- | --- |
| 32（未补齐） | 全部为 0 | 全部为 1 | 16 |
| 33（补一列） | 0～15 | 1～16 | 2 |
| 65（展开并补一列） | 0～15 | 1～16 | 2 |

因此本代码的 padding 将该读取的 **16 路冲突降到 2 路，而不是完全消除**。原因是一个 warp 覆盖两列、每列 16 个元素，两组 bank 有重叠。其他 warp 的分布只是平移，最大冲突度相同。

展开版第二次共享读取的偏移是 32 个 float，`32 % 32=0`，所以 bank 分布相同；两次读取是分别分析的访问，不能把两组数据合起来声称有 4 路冲突。

常见 32 × 32 方块转置的 padding 示例不能直接套用到这里的 32 × 16 线程映射。网页柱状图展示的是地址模型，不是 GPU 周期或性能实测。

## 6. 两份展开如何工作

`transposeSmemUnrollPad` 让每个块覆盖输入的 16 行 × 64 列：

```cpp
ix = 2 * 32 * blockIdx.x + tx;
iy = 16 * blockIdx.y + ty;
row_idx = ty * 65 + tx;
tile[row_idx]      = in[iy * nx + ix];
tile[row_idx + 32] = in[iy * nx + ix + 32];
```

物理每行有 65 个 float，其中 64 个数据、1 个 padding。同步之后：

```cpp
col_idx = icol * 65 + irow;
out[to]           = tile[col_idx];
out[to + ny * 32] = tile[col_idx + 32];
```

输入中向右 32 列，转置后变成输出中向下 32 行，所以输出地址偏移是 `ny*32`，不是 32。同一个线程执行两次加载、两次写出，一次块内同步。

## 7. 当前代码的正确性限制

这些是源码已有的问题，本文与网页只作解释，没有修改 CUDA 文件。

1. `transposeSmem` 用转换后的输出坐标判断整个加载、同步、写出代码块。对于不完整块，这不能代替输入加载边界检查，也不能保证所有线程一致到达同步点。
2. `transposeSmemUnpad` 的输出边界写成 `ix<nx && iy<ny`，正确输出范围应为 `ix<ny && iy<nx`。方阵会掩盖此问题；它同时具有上述边缘块问题。
3. 展开版用 `(ix+32)<nx && iy<ny` 把两次加载绑定在一起，会漏掉仅第一份有效的尾部；条件同步和按加载坐标约束写出也不适用于通用边缘块。
4. 展开版网格用 `ceil(nx/32)/2` 的整数截断。比如 nx=65，得到 1 个块，漏掉第 65 列；应按覆盖宽度计算 `(nx+63)/64`，并同时修正 kernel 中每份加载与写出的边界。
5. benchmark 的 4096 × 4096 方阵整除 64 和 16，恰好避开这些尺寸问题。打印 completed successfully 并不代表通过逐元素正确性验证。
6. `naive_transpose.cu` 的输出打印循环仍沿用输入行列数；当前方阵正常，推广到矩形时应按输出 nx 行、ny 列打印。

通用共享内存版本应采用以下结构（假设线程块与 tile 定义一致）：

```cpp
// 输入坐标与输出坐标分别计算并保留
if (input_x < nx && input_y < ny)
    tile[ty][tx] = in[input_y * nx + input_x];
__syncthreads();  // 所有线程无条件到达
if (output_x < ny && output_y < nx)
    out[output_y * ny + output_x] = tile[icol][irow];
```

有效输出对应的输入一定有效，因此此映射下不必为无效 tile 位置填零。展开版本需要对两份输入、两份输出分别判断，不能只修改 grid 就认为完成修复。

## 8. Benchmark 应如何解读

四个 wrapper 都使用 4096 × 4096 的 float 矩阵：5 次预热，随后用 CUDA event 包围 5 次 kernel 调用，总毫秒数除以 5。H2D 在计时前、D2H 在计时后，因此输出的是平均 kernel 时间，不是端到端耗时。

逻辑有效带宽（一次读 + 一次写，不代表实际显存流量）：

```text
GB/s = 2 * nx * ny * sizeof(float) / (平均毫秒数 * 10^6)
4096 × 4096 float：逻辑读写量 = 134217728 B = 128 MiB
```

当前没有保存运行结果，不应凭优化版本名称推断一定更快。比较前应：

- 用 CPU 参考式 `out[x*ny+y] == in[y*nx+x]` 校验全部元素，测试矩形、奇数尺寸和小于块的尺寸；现有共享版本需先修复边界。
- 启用并分别运行四个 wrapper，记录 GPU 型号、CUDA 版本、尺寸与时间；不要混入编造的数据。
- 检查每个 CUDA API 返回值及 kernel 执行错误。现有代码只在同步后调用一次 `cudaGetLastError()`，未逐项检查返回值。
- 正常路径销毁 `cudaEvent_t`；当前 wrapper 未调用 `cudaEventDestroy`，错误返回分支也未完整释放资源。
- 如果要单独比较算法因素，应留意基础版为 1024 线程/块，共享版本为 512 线程/块，且共享内存占用不同。

## 9. 使用网页

直接打开 [交互页面](transpose_visualizer.html)，或在本目录执行 `python -m http.server 8000`，再访问 `http://localhost:8000/transpose_visualizer.html`（服务器环境需端口转发）。

建议顺序：先用 4 × 2 教学块理解“加载的元素与写出的元素不同”，再切到 32 × 16 对照源码；切换 padding 观察 bank 分布，最后观察展开版的第二份数据如何从输入右侧移到输出下方。动画为索引模拟，阶段表示逻辑依赖，不模拟 GPU 实际调度。

参考：[NVIDIA 矩阵转置优化示例](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)、[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)。具体索引、冲突度和现有问题均以本目录代码分析为准。
