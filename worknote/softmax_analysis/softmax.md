# Softmax Benchmark 调用栈分析

使用 https://github.com/bytedance/xpu-perf 进行 softmax 前向性能分析的调用链路解析。=

命令: `source /home/sdp/fengqing/workspace/env.sh && python launch.py --backend INTEL --task softmax`

---

## 1. 入口 — launch.py

```
launch.py: __main__
├── parse_args()
│   ├── 扫描 backends/ 目录，构建 backend_list = ["INTEL", "GPU", ...]
│   ├── argparse 解析命令行参数: --backend INTEL, --task softmax
│   ├── importlib 动态加载 backends.INTEL.backend_intel.BackendINTEL
│   ├── backend_instance = BackendINTEL()
│   └── backend_instance.load_all_ops()  # 加载所有支持的op
│
├── parse_tasks(task_dir, "softmax")         → client.py
├── XpuPerfServer(engine_args_dict).__enter__()  → perf_engine.py
├── server_instance.normal_bench(test_cases)     → perf_engine.py
└── export_reports(report_dir, info_dict, ...)   → client.py
```

---

## 2. 任务解析 — client.py: parse_tasks()

```
parse_tasks(task_dir="workloads/basic", task="softmax")
├── rglob("*.json") 递归搜索所有json文件
├── 找到 workloads/basic/vector_norm_ops/softmax.json
├── parse_json_file() 解析JSON，展开参数组合
│   └── softmax.json 内容:
│       {
│         "cases": [{
│           "arg_type": "default",
│           "dtype": ["float32", "float16", "bfloat16"],
│           "batch_size": [1024],
│           "dim_size": [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
│         }]
│       }
│   → 展开为 3 × 11 = 33 个 test case
└── 返回 {"softmax": [case1, case2, ..., case33]}
```

---

## 3. 引擎映射 — core/ops/__init__.py

```
OP_ENGINE_MAPPING["softmax"] = "ComputeEngine"
DEFAULT_OP_IMPL_MAPPING["softmax"] = SoftmaxOp  (from core/ops/vector_norm_ops.py)
```

---

## 4. Server 启动 — perf_engine.py: XpuPerfServer

```
XpuPerfServer.__enter__()
└── create()
    └── 遍历 required_engines = {"ComputeEngine"}
        └── ComputeEngine(args_dict).start()
            └── mp.spawn() 为每个 device 创建子进程
                └── 子进程运行 compute_infer_loop()，等待 input_queue 任务
```

---

## 5. 任务分发与执行 — perf_engine.py → backend_intel.py

```
server_instance.normal_bench(test_cases)
├── OP_ENGINE_MAPPING["softmax"] → "ComputeEngine"
├── 将 softmax 的33个case分发给 ComputeEngine
└── ComputeEngine.dispatch(engine_test_cases)
    └── 子进程 compute_infer_loop() 从 input_queue 取出任务
        │
        ├── 对每个 case:
        │   ├── 实例化 SoftmaxOp(case_dict, backend_instance)
        │   ├── SoftmaxOp.prepare()                    [详见第6步]
        │   └── backend_instance.perf(op_instance)     [详见第7步]
        │
        └── 结果写入 output_queue
```

---

## 6. SoftmaxOp 准备 — core/ops/vector_norm_ops.py: SoftmaxOp.prepare()

```
SoftmaxOp.prepare()
├── 解析参数: dtype, batch_size, dim_size
├── 创建 input_tensor_info:
│   └── src: shape=[batch_size, dim_size], dtype=torch_dtype, device="xpu"
├── 创建 output_tensor_info:
│   └── dst: shape=[batch_size, dim_size], dtype=torch_dtype, device="xpu"
│
├── 计算IO字节数:
│   ├── input_tensor_size = batch_size × dim_size × dtype_size
│   ├── output_tensor_size = input_tensor_size  (softmax不改变shape)
│   ├── read_bytes  = input_tensor_size
│   ├── write_bytes = input_tensor_size
│   └── io_bytes    = read_bytes + write_bytes
│
└── 设置运行函数:
    └── _run_func = softmax_run
        └── torch.nn.functional.softmax(src, dim=-1)
```

**示例** (dtype=bfloat16, batch_size=1024, dim_size=131072):
- dtype_size = 2 bytes
- read_bytes  = 1024 × 131072 × 2 = 268,435,456 B
- write_bytes = 268,435,456 B
- io_bytes    = 536,870,912 B

---

## 7. 性能测量 — backend_intel.py: perf() → core_perf()

```
BackendINTEL.perf(op_instance)  → 实际调用 run_perf(self, op_instance)
│
├── 计算可用显存，决定 max_data_cnt (可同时驻留的tensor副本数)
├── op_instance.create_tensors(max_data_cnt) 创建输入tensor
├── random.shuffle(tensor_list) 打乱顺序(避免cache命中)
│
├── 第一次 core_perf(): warmup=2, iters=2, profiling=False
│   └── 粗测 latency，用于计算合理的迭代次数
│
├── prefer_iters = max(1e6 / latency_us, 10)  # 至少10次，目标总时长约1秒
│
└── 第二次 core_perf(): warmup=2, iters=prefer_iters, profiling=True
    │
    ├── [profiling路径] torch.profiler.profile():
    │   ├── schedule: wait=0, warmup=warmup_iters, active=prefer_iters
    │   ├── 每次迭代:
    │   │   ├── op_instance.core_run(tensor_mapping)
    │   │   │   └── torch.nn.functional.softmax(src, dim=-1)
    │   │   ├── torch.xpu.synchronize()
    │   │   └── prof.step()
    │   ├── export_chrome_trace("profiling/<pid>/trace.json")
    │   └── 解析 trace.json:
    │       ├── 提取 cat=="kernel" 或 "gpu_memcpy" 的事件
    │       ├── 累计每个 kernel 的耗时
    │       └── average_latency = total_kernel_time / prefer_iters
    │
    └── [非profiling路径] torch.xpu.Event 计时:
        ├── start_event.record()
        ├── 循环 prefer_iters 次 core_run()
        ├── end_event.record() + synchronize()
        └── latency_us = elapsed_time * 1e3 / prefer_iters
```

---

## 8. 结果汇总 — SoftmaxOp.summary()

```
op_instance.summary(latency_us, kernel_mapping)
└── 返回 target_dict:
    {
      "latency(us)":            latency_us,
      "read_bytes(B)":          read_bytes,
      "write_bytes(B)":         write_bytes,
      "io_bytes(B)":            io_bytes,
      "mem_bw(GB/s)":           io_bytes / latency_us / 1e3,
      "calc_flops":             0,
      "calc_flops_power(tflops)": 0.0,
      "calc_mem_ratio":         0.0,
      "kernels":                [kernel_name1, ...]
    }
```

**核心公式:**

```
mem_bw(GB/s) = io_bytes(B) / latency(μs) / 1000
```

**示例** (bfloat16, 1024×131072):
```
io_bytes    = 536,870,912 B
latency     = 3639.454 μs
mem_bw      = 536870912 / 3639.454 / 1000 = 147.514 GB/s
```

---

## 9. 报告导出 — client.py: export_reports()

```
export_reports(report_dir, info_dict, test_cases, bench_results)
├── 创建目录: reports/INTEL/<device_name>/
├── 写入 info.json (设备信息、环境信息)
│
├── 遍历每个 op 的结果:
│   ├── 按 provider 分组
│   └── 对每个 provider:
│       ├── 创建目录: reports/INTEL/<device_name>/softmax/torch/
│       ├── 写入 softmax-torch.jsonl (每行一个JSON)
│       └── 写入 softmax-torch.csv
│
└── 最终文件路径:
    reports/INTEL/Intel(R) Arc(TM) Pro B60 Graphics/softmax/torch/softmax-torch.jsonl
```

---

## 完整调用链路图

```
launch.py main
│
├─ parse_tasks("workloads/basic", "softmax")     # 解析 softmax.json → 33个case
│
├─ XpuPerfServer.__enter__()
│   └─ ComputeEngine.start()
│       └─ mp.spawn → compute_infer_loop()       # 子进程就绪
│
├─ normal_bench(test_cases)
│   └─ ComputeEngine.dispatch()
│       └─ 子进程循环处理每个case:
│           ├─ SoftmaxOp(case).prepare()          # 构建tensor、计算io_bytes
│           ├─ BackendINTEL.perf()
│           │   ├─ create_tensors()               # 在XPU上分配tensor
│           │   ├─ core_perf(warmup)              # 粗测latency
│           │   └─ core_perf(profiling)           # 精确测量
│           │       └─ softmax(src, dim=-1)       # 实际kernel执行
│           └─ summary() → {latency, mem_bw, ...} # 汇总结果
│
└─ export_reports()                               # 输出 .jsonl + .csv
```

---

# CUDA Softmax Forward — inner_size=1 时的 Kernel 选择策略

> 源码: `aten/src/ATen/native/cuda/SoftMax.cu` → `host_softmax()`
> 条件: softmax 作用在最后一个维度（inner_size=1）

### 总体分派流程

```
dim_size ≤ 2048 且 dim_size*sizeof(scalar_t) ≤ 8192?
  ├─ YES → softmax_warp_forward (warp-level, softmax_warp_forward.cuh)
  └─ NO  → 三级 Kernel 选择 (threadblock-level)
```

### 三级 Kernel 选择（dim_size > 2048 的非 fast_softmax 路径）

#### 关键变量

```cpp
constexpr int ILP = sizeof(float4) / sizeof(scalar_t);
// ILP = 向量化宽度 = 16/sizeof(scalar_t)
//   float32 → ILP=4,  float16/bfloat16 → ILP=8

dim3 block = SoftMaxForward_getBlockSize(dim_size);
// block.x = min(dim_size, 1024)，向上取 warp_size(32) 的倍数

int32_t potential_reg_cnt = (dim_size + block.x - 1) / block.x;
// = 每个线程需要负责的元素数
```

#### 第一优先：Register 路径 — `cunn_SoftMaxForwardReg`

```
条件: potential_reg_cnt < 10 (即 dim_size < block.x × 10 ≈ 10240)
```

| 特征 | 说明 |
|---|---|
| 数据缓存位置 | **寄存器** (`scalar_t reg[reg_cnt]`) |
| Global Memory 读取次数 | **1 次** |
| 向量化 | **否**，逐标量 `input[threadIdx.x + reg_idx * blockDim.x]` |
| 模板参数 `reg_cnt` | 编译期常量 1~9，通过 switch-case 分派 |
| Reduction | `blockReduceWarp`（warp shuffle → smem） |

**工作流程:**
1. 每个线程读 `reg_cnt` 个元素到寄存器数组 `reg[]`
2. 第一遍扫描 `reg[]` 求 `threadMax` → `blockReduceWarp` 得 `max_k`
3. 第二遍扫描 `reg[]` 求 `threadExp` (sum of exp) → `blockReduceWarp` 得 `sumAll`
4. 第三遍扫描 `reg[]`，`output[offset] = epilogue(reg[reg_idx])` 写回

**优点:** 数据常驻寄存器，三个 pass 零额外访存
**限制:** `reg_cnt` 必须是编译期常量（模板参数），所以只能枚举 1~9

**适用 dim_size 范围 (block.x=1024):**

| reg_cnt | dim_size 范围 |
|---|---|
| 1 | 1 ~ 1024 (不可能，走了 softmax_warp_forward) |
| 2 | 1025 ~ 2048 (不可能，走了 softmax_warp_forward) |
| 3 | 2049 ~ 3072 |
| 4 | 3073 ~ 4096 |
| 5 | 4097 ~ 5120 |
| 6 | 5121 ~ 6144 |
| 7 | 6145 ~ 7168 |
| 8 | 7169 ~ 8192 |
| 9 | 8193 ~ 9216 |

> 注: 实际 dim_size ≤ 2048 且 bytes ≤ 8192 时走 softmax_warp_forward。
> 对于 float16/bfloat16: dim_size ≤ 4096(bytes=8192) 走 softmax_warp_forward。
> 所以 Reg 路径对 fp16/bf16 从 dim_size=4097 开始生效。

#### 第二优先：Shared Memory 路径 — `cunn_SoftMaxForwardSmem`

```
条件（全部满足）:
  1. dim_size < (sharedMemPerBlock - smem_reduction_sz) / sizeof(scalar_t)
     — shared memory 能装下整行
  2. input_ptr % 16 == 0   — 输入地址 16 字节对齐
  3. output_ptr % 16 == 0  — 输出地址 16 字节对齐
  4. dim_size % ILP == 0   — dim_size 是向量宽度的倍数
```

| 特征 | 说明 |
|---|---|
| 数据缓存位置 | **Shared Memory** |
| Global Memory 读取次数 | **1 次** |
| 向量化 | **是**，`aligned_vector<scalar_t, ILP>` (float4 = 16B) |
| Reduction | `blockReduceWarp`（warp shuffle → smem） |

**工作流程:**
1. 向量化读取 global memory → 同时存入 smem 缓存 + 求 `max_k`
2. 从 smem 向量化读取，计算 `sum_exp` → `blockReduceWarp` 得 `sumAll`
3. 从 smem 向量化读取，`epilogue()` 计算后向量化写回 global memory

**适用场景:**
- `dim_size` 在 ~9217 到 ~12288（48KB smem / 4B per float32 = 12288）
- 或 fp16 时最多 ~24576（48KB / 2B）
- 且地址对齐 + dim_size 整除 ILP

#### 第三兜底：Global Memory 路径 — `cunn_SoftMaxForward`

```
条件: 以上都不满足时的 fallback
```

| 特征 | 说明 |
|---|---|
| 数据缓存位置 | **无缓存** |
| Global Memory 读取次数 | **3 次**（max、sumexp、output 各读一次） |
| 向量化 | **是**，`ilpReduce` 用 `aligned_vector<T, ILP>` 向量化读 |
| 对齐优化 | 若 input/output 对齐相同 → `WriteFpropResultsVectorized`（向量化写） |

**工作流程:**
1. `ilpReduce<MaxFloat>` — 向量化读 global memory 求 `max_k`
2. `ilpReduce<SumExpFloat>` — 再读一次 global memory 求 `sumAll`
3. `WriteFpropResultsVectorized` / `WriteFpropResults` — 第三次读 + 写回

**对齐处理（ilpReduce 细节）:**
```
数据:  [.....|AAAA|BBBB|CCCC|...|tail]
        ↑ shift              向量化读取区域        ↑ epilogue
```
- `shift = input_ptr % 16 / sizeof(scalar_t)` — 对齐偏移
- 先逐标量处理 shift 个不对齐元素
- 然后向量化批量读取（ILP 个元素一次 = float4 = 128 bit）
- 尾部不足 ILP 的再逐标量处理

### 完整路径总结

```
host_softmax(input, dim)
│
├─ inner_size == 1 (softmax on last dim)
│   │
│   ├─ dim_size ≤ 2048 && dim_size*sizeof(T) ≤ 8192
│   │   └─ dispatch_softmax_forward() → softmax_warp_forward (warp-level)
│   │      • 一个 warp 处理一行，warp shuffle reduce
│   │      • 模板参数 log2_elements 0~11
│   │
│   └─ dim_size > 2048 || dim_size*sizeof(T) > 8192
│       │
│       └─ [CUDA] use_fast_softmax=false
|           |  一个 block 处理一行，blockDim.x ≤ 1024
│           ├─ reg_cnt < 10      → cunn_SoftMaxForwardReg   (寄存器缓存, 读1次)
│           ├─ can_use_smem      → cunn_SoftMaxForwardSmem  (smem缓存, 读1次, 向量化)
│           └─ else              → cunn_SoftMaxForward      (无缓存, 读3次, 向量化)
│
└─ inner_size > 1 (softmax on non-last dim)
    └─ cunn_SpatialSoftMaxForward (2D grid, 沿 inner_size 并行)
```

### half_to_float 的影响

当 `half_to_float=true`（输入 fp16，输出 fp32）时:
- softmax_warp_forward 阈值更保守: `dim_size ≤ 1024 && bytes ≤ 4096`
- Reg 路径**不可用**（仅 `!half_to_float` 分支有 Reg 路径）
- 只有 Smem 和 Gmem 两级选择

---

# XPU Softmax Forward — inner_size=1 时的 Kernel 选择策略

> 源码: `third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/SoftMaxKernels.cpp`
> 函数: `spatial_softmax_forward()` → 两级 kernel 选择
> 条件: softmax 作用在最后一个维度（inner_size=1），每个 work-group 处理一个或多个 batch 行

### 关键常量

```cpp
constexpr int max_vec_size = sizeof(float) * 4 / sizeof(inscalar_t);
// float32 → 4,  float16/bfloat16 → 8

constexpr int INNER_LOOP = max_vec_size * 2;
// float32 → 8,  float16/bfloat16 → 16
// 每个 work-item 处理的最大元素数
```

| 数据类型 | `max_vec_size` | `INNER_LOOP` |
|---|---|---|
| float32 | 4 | 8 |
| float16 | 8 | 16 |
| bfloat16 | 8 | 16 |

### 预处理：三个决策

**1. 对齐检查**
```cpp
int input_start = ((uint64_t)input_ptr) % align_bytes / sizeof(inscalar_t);
int output_start = ((uint64_t)output_ptr) % align_bytes / sizeof(outscalar_t);
// input_start == 0 && output_start == 0 → 16字节对齐
```

**2. 索引类型**
```cpp
bool can_use_32bit_index = canUse32BitIndexMath(input) && canUse32BitIndexMath(output);
// tensor < 4GB → uint32_t，否则 uint64_t
```

**3. SIMD 宽度选择**
```cpp
int SIMD = dev_prop->sub_group_sizes[1];  // 设备支持的最大子组宽度 (32 或 16)
if (SIMD == SIMD32 && dim_size < SIMD16 * INNER_LOOP) {
    SIMD = SIMD16;  // 小 dim_size 降级到 SIMD16
}
```
降级阈值: fp32 → `dim_size < 128`, fp16/bf16 → `dim_size < 256`

### 总体分派流程

```
inner_size == 1?
├─ YES → can_use_32bit_index?
│         ├─ YES → 尝试快速路径 (dispatch_softmax_forward_kernel)
│         │         ├─ 成功 (maxWGSize * INNER_LOOP >= dim_size) → 完成
│         │         └─ 失败 (dim_size 太大) → use_slow_path = true
│         └─ NO  → use_slow_path = true
│
│        use_slow_path?
│         ├─ YES → 慢速路径 (softmax_forward_kernel)
│         └─ NO  → 已完成
│
└─ NO  → SpatialSoftmaxForwardKernelFunctor (3D grid)
```

### 第一优先：快速路径 — `DispatchSoftmaxForwardKernelFunctor`

**进入条件**: `can_use_32bit_index` 且 `maxWGSize * INNER_LOOP >= dim_size`
- fp32: dim_size ≤ ~8192 (maxWGSize=1024, INNER_LOOP=8)
- fp16/bf16: dim_size ≤ ~16384 (maxWGSize=1024, INNER_LOOP=16)

**特征**:
- 数据全部加载到**寄存器** `vec_t reg_in[outer_loop]`
- Global Memory **读 1 次**
- 使用 `softmax_group_reduce`（`shift_group_left` 子组内归约 + SLM 跨子组归约）
- 显式向量化 `aligned_vector<inscalar_t, vec_size>`
- `SYCL_REQD_SUB_GROUP_SIZE(SIMD)` 编译期固定子组宽度

**工作流程**:
1. 向量化加载到 `reg_in[outer_loop]` + 求 `max_value`
2. `softmax_group_reduce<SIMD>(max)` → 全局 max
3. 扫描 `reg_in[]` 求 `sum_value = Σexp(x - max)`
4. `softmax_group_reduce<SIMD>(sum)` → 全局 sum
5. 扫描 `reg_in[]` 计算 `exp(x-max)/sum` 写回

**SIMD32 分支** (设备支持 SIMD32 且 dim_size ≥ SIMD16 * INNER_LOOP):

| 条件 | vec_size | outer_loop | 每 WI 处理元素数 | 每子组覆盖 |
|---|---|---|---|---|
| 对齐 + dim%vec==0 | max_vec_size | INNER_LOOP/max_vec_size = 2 | 2×vec | 32×2×vec |
| 不对齐 | 1 | INNER_LOOP | INNER_LOOP | 32×INNER_LOOP |

fp32 示例: 对齐时 vec=4, outer_loop=2, 每 WI 处理 8 个 fp32, 每子组覆盖 256 个元素
fp16 示例: 对齐时 vec=8, outer_loop=2, 每 WI 处理 16 个 fp16, 每子组覆盖 512 个元素

**SIMD16 分支** (设备仅支持 SIMD16，或 dim_size < SIMD16 * INNER_LOOP):

| 条件 | vec_size | outer_loop | 每 WI 处理 | 说明 |
|---|---|---|---|---|
| 对齐 + dim≤4×SIMD(=64) + vec≥4 | 4 | 1 | 4 | 极小 dim，减小 vec 避免浪费 |
| 对齐 + dim≤vec×SIMD | max_vec_size | 1 | max_vec | 一个子组一次 outer_loop 即可覆盖 |
| 对齐 + dim较大 | max_vec_size | INNER_LOOP/vec×2 = 4 | vec×4 | 标准路径，outer_loop 翻倍补偿 SIMD 减半 |
| 不对齐 | 1 | INNER_LOOP×2 | IL×2 | 标量版，outer_loop 翻倍 |

> 关键: SIMD16 的 outer_loop 是 SIMD32 的 **2 倍**。
> 原因: SIMD32 有 32 个 WI 参与，SIMD16 只有 16 个，所以每个 WI 需要多处理 2× 的数据才能覆盖相同 dim_size。
> 寄存器方面: SIMD16 每个 WI 虽然 outer_loop 翻倍，但实际占用的寄存器槽位更少（SIMD16 一条寄存器只 serve 16 个 lane vs SIMD32 serve 32 个 lane）。

**`get_wgroup_size` 的 work-group 尺寸决策**:

```cpp
// 每个 WI 处理 outer_loop × vec_size 个元素
// 需要 local_size = ceil(dim_size / (outer_loop × vec_size)) 个 WI
local_size = min(ceil(dim_size / (outer_loop * vec_size)), maxWGSize);
sub_group_num = ceil(local_size / SIMD);
local_size_col = sub_group_num * SIMD;

// 特殊情况: 一个 WI 就能覆盖整行
if (dim_size <= vec_size * outer_loop) {
    local_size_col = 1;       // 一个 WI 处理一行
    local_size_row = SIMD;    // 一个 work-group 处理 SIMD 行
}

// 合并行处理: outer_size 很大时，一个 work-group 处理多行
while (global_size_row/2 > MIN_WG_NUM(32768) &&
       local_size_row*2 * local_size_col <= maxWGSize &&
       global_size_row % 2 == 0) {
    local_size_row *= 2;
    global_size_row /= 2;
}
```

**快速路径失败条件**: `maxWGSize * INNER_LOOP < dim_size` → 返回 false → 走慢速路径

### 第二兜底：慢速路径 — `SoftmaxForwardKernelFunctor`

**条件**: 快速路径返回 false（dim_size 太大）或不能用 32 位索引

| 特征 | 说明 |
|---|---|
| 数据缓存 | **无**，每 pass 从 global memory 读取 |
| Global Memory 读取次数 | **3 次**（max、sum、output 各读一次） |
| 向量化 | **是**，`aligned_vector<inscalar_t, vec_size>` |
| Reduction | `sycl::reduce_over_group`（整个 work-group 级别） |
| SIMD 约束 | **无**，不使用 `SYCL_REQD_SUB_GROUP_SIZE` |
| 对齐处理 | 通过 `start` 偏移量内部处理不对齐 |
| work-group 大小 | `min(ceil(dim_size/vec_size), maxWGSize)` |

**分派**:
```
can_use_32bit_index?
├─ YES: input_start == output_start?
│       ├─ YES → vec_size = max_vec_size (向量化)
│       └─ NO  → vec_size = 1 (标量)
└─ NO:  input_start == output_start?
        ├─ YES → vec_size = max_vec_size, IndexType = uint64_t
        └─ NO  → vec_size = 1, IndexType = uint64_t
```

**对齐处理逻辑**:
```
start = input_ptr % align_bytes / sizeof(scalar_t)
// 向量化读取时，地址前移 start 到对齐边界
// 首尾用 linear_idx 边界检查保护
// 中间部分直接向量化读+写
```

**工作流程**:
1. 向量化循环读 global memory → 求 `max_value` → `reduce_over_group(maximum)`
2. 再次向量化循环读 global memory → 求 `sum_value` → `reduce_over_group(plus)`
3. 第三次向量化循环读 global memory → `exp(x-max)/sum` 或 `x-max-log(sum)` → 写回

### 完整路径总结

```
spatial_softmax_forward(input, dim)
│
├─ inner_size == 1 (softmax on last dim)
│   │
│   ├─ can_use_32bit_index → 尝试快速路径
│   │   │
│   │   ├─ SIMD32 (设备支持 && dim_size ≥ SIMD16*INNER_LOOP)
│   │   │   ├─ 对齐 + dim%vec==0 → vec=max_vec, outer_loop=2      (向量化)
│   │   │   └─ else             → vec=1, outer_loop=INNER_LOOP     (标量)
│   │   │
│   │   └─ SIMD16 (设备仅支持 || dim_size < SIMD16*INNER_LOOP)
│   │       ├─ 对齐 + dim≤64     → vec=4, outer_loop=1             (极小dim)
│   │       ├─ 对齐 + dim≤vec*16 → vec=max_vec, outer_loop=1       (小dim)
│   │       ├─ 对齐 + dim较大    → vec=max_vec, outer_loop=4       (标准)
│   │       └─ 不对齐            → vec=1, outer_loop=INNER_LOOP*2  (标量)
│   │
│   │   → DispatchSoftmaxForwardKernelFunctor
│   │     • 寄存器缓存 reg_in[outer_loop]
│   │     • Global Memory 读 1 次
│   │     • softmax_group_reduce (shift_group_left + SLM)
│   │     • 成功条件: maxWGSize * INNER_LOOP >= dim_size
│   │
│   └─ 快速路径失败 或 不能用32位索引 → 慢速路径
│       │
│       ├─ input_start == output_start → vec=max_vec (向量化)
│       └─ else                        → vec=1 (标量)
│
│       → SoftmaxForwardKernelFunctor
│         • 无缓存
│         • Global Memory 读 3 次
│         • reduce_over_group (work-group 级别)
│
└─ inner_size > 1 (softmax on non-last dim)
    └─ SpatialSoftmaxForwardKernelFunctor (3D grid)
```

### XPU vs CUDA 对比

| 维度 | XPU | CUDA |
|---|---|---|
| 快速路径上限 (fp32) | ~8192 (maxWGSize×INNER_LOOP) | 9216 (Reg) 或 2048 (Persistent) |
| 快速路径上限 (fp16) | ~16384 | 4096 (Persistent) 或 ~9216 (Reg) |
| 快速路径缓存 | 寄存器 `reg_in[outer_loop]` | 寄存器 `reg[reg_cnt]` (Reg) / warp寄存器 (Persistent) |
| 中间缓存层级 | **无** | 有 Shared Memory 路径 (`cunn_SoftMaxForwardSmem`) |
| 慢速路径 | 读 3 次 gmem, `reduce_over_group` | 读 3 次 gmem, `blockReduceWarp` |
| Reduction 方式 | `shift_group_left` + SLM (快), `reduce_over_group` (慢) | warp shuffle + smem |
| SIMD 自适应 | SIMD32 ↔ SIMD16 动态选择 | 固定 warp=32 |
| 向量化 | 显式 `aligned_vector`，快/慢路径都用 | Reg 路径无向量化，Smem/Gmem 路径用 `aligned_vector` |
| 行合并处理 | `local_size_row` 机制，一个 WG 处理多行 | 无，严格一个 block 一行 |

**关键差异**:
1. XPU **没有** CUDA 的 shared memory 缓存中间层。快速路径失败后直接跳到读 3 次 global memory 的兜底路径
2. XPU 通过 `local_size_row` 可以让一个 work-group 处理多行 (当 outer_size 很大且 dim_size 小时)，提高 WG 利用率
3. XPU 的快速路径对 fp16/bf16 上限更高 (16384 vs CUDA 的 4096/9216)，因为 INNER_LOOP 更大

---

# Softmax Slow Path (`SoftmaxForwardKernelFunctor`) Vec Store 优化 — Unitrace Profiling 实验

## 实验目的

对比 PyTorch main branch 与添加 softmax vec store 优化后的版本，通过 Unitrace 硬件计数器抓取分析 store 写放大问题的改善效果。

## 实验环境

- **设备**: Intel Arc Pro B60 Graphics
- **驱动/环境**: oneAPI + PTI Unitrace 2.3.0
- **Python**: 3.10 (conda env: lfq)
- **Benchmark 脚本**: `temp/softmax_bench.py`
- **两个 PyTorch whl**:
  - Main branch: `pytorch/main_dist/torch-2.13.0a0+git0290cd4-cp310-cp310-linux_x86_64.whl`
  - Vec store 优化: `pytorch/dist/torch-2.13.0a0+git0290cd4-cp310-cp310-linux_x86_64.whl`

## Benchmark 说明

`softmax_bench.py` 对以下配置运行 `F.softmax(src, dim=-1)`:

| Shape | Dtype |
|---|---|
| [1024, 32768] | float16, float32 |
| [1024, 65536] | float16, float32 |
| [1024, 131072] | float16, float32 |

使用 `--profile` 模式: warmup + flush cache + 1 iteration，适合 Unitrace 抓取。

## 实验流程

### 1. 环境准备

```bash
# 初始化 oneAPI 和 conda 环境
source /home/sdp/fengqing/workspace/env.sh

# 设置 observation_paranoid 允许非 root 抓取 metrics
echo 0 | sudo tee /proc/sys/dev/xe/observation_paranoid

# Unitrace 路径
UNITRACE=/home/sdp/yifeng/pti-gpu/tools/unitrace/build/unitrace
```

### 2. 创建工作目录

```bash
mkdir -p temp/softmax_unitrace/{main,vecstore}
```

### 3. Main branch profiling

```bash
# 安装 main branch whl
pip install pytorch/main_dist/torch-2.13.0a0+git0290cd4-cp310-cp310-linux_x86_64.whl \
    --force-reinstall --no-deps

# 分别抓取 3 个 metric group
cd temp/softmax_unitrace/main

$UNITRACE -q -g ComputeBasic -o compute_basic.csv \
    python temp/softmax_bench.py --profile

$UNITRACE -q -g DeviceCacheProfile -o cache.csv \
    python temp/softmax_bench.py --profile

$UNITRACE -q -g MemoryProfile -o memory.csv \
    python temp/softmax_bench.py --profile
```

### 4. Vec store 优化版 profiling

```bash
# 安装 vec store 优化版 whl
pip install pytorch/dist/torch-2.13.0a0+git0290cd4-cp310-cp310-linux_x86_64.whl \
    --force-reinstall --no-deps

# 同样抓取 3 个 metric group
cd temp/softmax_unitrace/vecstore

$UNITRACE -q -g ComputeBasic -o compute_basic.csv \
    python temp/softmax_bench.py --profile

$UNITRACE -q -g DeviceCacheProfile -o cache.csv \
    python temp/softmax_bench.py --profile

$UNITRACE -q -g MemoryProfile -o memory.csv \
    python temp/softmax_bench.py --profile
```

### 5. Unitrace 参数说明

| 参数 | 含义 |
|---|---|
| `-q` | Metric query 模式，每个 kernel instance 输出一行聚合计数器 |
| `-g <group>` | 指定 metric group，每次只能抓一个 group |
| `-o <prefix>` | 输出 CSV 文件前缀，实际文件名为 `<prefix>.metrics.<pid>.csv` |

### 6. 结果文件

```
temp/softmax_unitrace/
├── main/
│   ├── compute_basic.metrics.<pid>.csv    # ComputeBasic group
│   ├── cache.metrics.<pid>.csv            # DeviceCacheProfile group
│   └── memory.metrics.<pid>.csv           # MemoryProfile group
└── vecstore/
    ├── compute_basic.metrics.<pid>.csv
    ├── cache.metrics.<pid>.csv
    └── memory.metrics.<pid>.csv
```

## 实验结果

### FP16 Softmax 性能对比

| Shape | Main GpuTime | VecStore GpuTime | 加速比 |
|---|---|---|---|
| [1024, 32768] | 1.00 ms | 0.54 ms | **1.84x** |
| [1024, 65536] | 1.61 ms | 0.82 ms | **1.96x** |
| [1024, 131072] | 3.40 ms | 1.79 ms | **1.90x** |

### FP32 Softmax 性能对比

| Shape | Main GpuTime | VecStore GpuTime | 加速比 |
|---|---|---|---|
| [1024, 32768] | 0.72 ms | 0.73 ms | 1.00x |
| [1024, 65536] | 1.60 ms | 1.61 ms | 1.00x |
| [1024, 131072] | 4.94 ms | 4.93 ms | 1.00x |

### FP16 关键 Metrics 变化 (以 [1024, 131072] 为例)

**ComputeBasic group**:

| Metric | Main | VecStore | 变化 |
|---|---|---|---|
| GpuTime [ns] | 3,402,776 | 1,789,476 | -47.4% |
| XVE_ACTIVE [%] | 31.5 | 56.9 | +80.5% |
| XVE_STALL [%] | 66.8 | 42.4 | -36.5% |
| XVE_SHARED_FUNCTION_ACCESS_HOLD [%] | 54.5 | 10.4 | -81.0% |
| L3_WRITE [events] | 16,777,224 | 4,194,313 | **-75.0%** |
| L3_STALL [%] | 69.0 | 1.2 | **-98.3%** |
| GPU_MEMORY_BYTE_READ [bytes] | 606,532,608 | 338,324,480 | -44.2% |
| GPU_MEMORY_BYTE_WRITE [bytes] | 277,770,752 | 277,636,608 | -0.0% |

**DeviceCacheProfile group**:

| Metric | Main | VecStore | 变化 |
|---|---|---|---|
| LOAD_STORE_CACHE_L3_WRITE [events] | 16,777,216 | 4,194,304 | **-75.0%** |
| L3_SUPERQ_FULL [%] | 10.1 | 0.1 | **-99.0%** |
| L3_INPUT_AVAILABLE [%] | 72.2 | 8.5 | -87.7% |
| GPU_MEMORY_L3_READ [events] | 5,523,634 | 1,321,739 | -76.1% |

**MemoryProfile group**:

| Metric | Main | VecStore | 变化 |
|---|---|---|---|
| XVE_LSC_WRITE_MESSAGE_COUNT | 2,097,152 | 524,288 | **-75.0%** |
| XVE_LSC_REGISTER_REQUEST_COUNT | 19,464,192 | 13,172,736 | -32.3% |
| LOAD_STORE_CACHE_ACCESS [events] | 29,392,896 | 16,809,984 | -42.8% |
| GPU_MEMORY_64B_TRANSACTION_READ | 4,749,661 | 2,724,987 | -42.6% |

### FP32: 无显著变化

FP32 的所有 metrics 在两个版本间变化均在 ±2% 以内，符合预期 — FP32 d32 store 本身就是 full CL write，不存在 partial write 问题。

## 分析

详细的 counter 对比和 ASM 分析见 `temp/softmax_vec_store_counters.md`。

### 根因: FP16 4×d32 partial write（ASM 确认）

通过将 `SoftmaxForwardKernelFunctor` 提取为独立 `.cpp` 编译并 dump ASM
（`icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2` + `IGC_ShaderDumpEnable=1`），
确认了 main branch 热路径的实际 store 指令。

**之前的错误认知**: 热路径使用 `d16u32` store，通过 byte-lane 拆分产生写放大。

**ASM 实际情况**: 热路径使用 **4 条 `store.ugm.d32.a64`**（编译器将 2 个 half 打包为 1 个 dword），
不是 `d16u32`。`d16u32` 仅出现在边界 fallback 路径（处理 start/remaining 不对齐），正常对齐数据不会走到。

```
main 热路径:     4× store.ugm.d32.a64   [r24:4], [r24:4+0x4], [r24:4+0x8], [r24:4+0xC]
vecstore 热路径: 1× store.ugm.d32x4.a64 [r4:4]   r8:8
```

4 条独立的 d32 store 各写 cache line 的 1/4（每条 4B × 32 lanes = 128B），
L3 收到 4 次 partial write → 第一次触发 RFO (Read-For-Ownership) → 额外 DRAM 读。
vecstore 用 1 条 d32x4 一次写满整个 CL（16B × 32 lanes = 512B），消除 RFO。

### 为什么 IGC 不自动合并 FP16 的 4×d32？

FP32 main branch 的热路径 **已经是 `store.ugm.d32x4.a64`** — IGC 自动完成了 store merging。
FP16 的 4×d32 没有被合并，原因是 **half→dword pack 导致源寄存器不连续**。

FP16 每次写之前需要 `float→half` 转换 + 把 2 个 half 塞进 1 个 dword 的高低 16-bit：
```asm
mov (32|M0)  r9.0<2>:hf    acc0.0<1;1,0>:f      // float→half, stride-2 写入
mov (32|M0)  r32.1<2>:uw   r9.0<2;1,0>:uw       // pack 到 dword 高 16-bit
...
store.ugm.d32.a64 (32|M0)  [r24:4]      r4:2    // 源: r4
store.ugm.d32.a64 (32|M0)  [r24:4+0x4]  r32:2   // 源: r32（不连续！）
store.ugm.d32.a64 (32|M0)  [r24:4+0x8]  r12:2   // 源: r12
store.ugm.d32.a64 (32|M0)  [r24:4+0xC]  r6:2    // 源: r6
```

4 条 store 的源寄存器 r4, r32, r12, r6 完全不连续。
IGC 合并为 d32x4 需要源数据在连续 8 个寄存器中（如 r8:8），但 half→dword pack 把数据分散了。

FP32 不存在这个问题：计算直接在 float 上，结果自然排列在连续寄存器中，IGC 能直接合并。

| | FP16 main | FP16 vecstore | FP32 main | FP32 vecstore |
|---|---|---|---|---|
| 热路径 store | 4× d32 | **1× d32x4** | **1× d32x4** | 1× d32x4 |
| IGC 能 merge？ | 不能（pack 打断） | 源码显式 | 能（寄存器连续） | 源码显式 |
| partial write？ | 是 → RFO | 否 | 否 | 否 |

**本质**: vec store patch 解决的是 IGC 对 FP16 store merging 的缺陷 —
half→dword pack 导致寄存器不连续，阻止了自动合并。FP32 不需要这个 patch。

### 完整因果链

```
main: 4× store.ugm.d32.a64（每条写 CL 的 1/4）
  → LSC 发出 4 条 store message（LSC WRITE_MESSAGE 4x）
    → L3 收到 4 次 partial write（L3_WRITE 4x）
      → 第一次 partial write 时 CL 不在 L3 → RFO（GPU_MEMORY_L3_READ +75~80%）
        → 额外 DRAM 读（GPU_MEMORY_BYTE_READ +43~50%）
      → RFO 读入的 CL 被后续 3 次 partial write 命中（L3_HIT 虚高）
      → 大量写请求塞满 L3 输入队列（L3_SUPERQ_FULL 12%）
        → L3 bank stall（L3_STALL 70%）
          → 背压传导到 LSC → XVE（XVE_SF_HOLD 55%）
            → XVE 利用率下降（XVE_ACTIVE 31%）

vecstore: 1× store.ugm.d32x4.a64（一次写满整个 CL）
  → LSC 发出 1 条 store message
    → L3 收到 1 次 full write → 直接 allocate + fill，无 RFO
      → 以上所有连锁反应全部消除
```

### Vec store 优化效果

- **L3 WRITE 减少 75%**: 4×partial write → 1×full write
- **RFO 读消除**: GPU_MEMORY_BYTE_READ 减少 ~50%，GPU_MEMORY_L3_READ delta 精确等于 output CL 数量
- **L3 拥塞消失**: L3_STALL 从 69% → 1.2%，L3_SUPERQ_FULL 从 10% → 0%
- **XVE 利用率提升**: XVE_ACTIVE 从 31% → 57%，SF_HOLD 从 55% → 10%
- **整体加速 ~1.9x**

### ASM 文件位置

```
temp/softmax_asm/
├── main_softmax.cpp       # main branch 独立编译源码
├── vecstore_softmax.cpp   # vecstore branch 独立编译源码
├── main_fp16.asm          # main FP16 kernel ASM (1718 行)
├── vecstore_fp16.asm      # vecstore FP16 kernel ASM (1822 行)
├── main_fp32.asm          # main FP32 kernel ASM (1193 行)
└── vecstore_fp32.asm      # vecstore FP32 kernel ASM (1246 行)
```
