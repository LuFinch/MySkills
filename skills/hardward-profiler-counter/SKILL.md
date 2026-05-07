# Hardware Profiler Counter Skill

## Overview

This skill covers using **Unitrace** (Intel XPU) and **Nsight Compute** (NVIDIA CUDA) to collect GPU hardware performance counters, analyze kernel bottlenecks, and understand GPU execution state.

---

## Part 1: Unitrace (Intel XPU)

### 1.1 What is Unitrace

Unitrace is a unified tracing and profiling tool from Intel PTI (Performance Tools Interface). It can:
- Trace host/device API calls and kernel execution timelines
- Query hardware performance metrics per kernel instance (`-q`)
- Sample hardware metrics in time-based mode (`-k`)
- Sample EU stalls at instruction level (`--stall-sampling`)

Source: https://github.com/intel/pti-gpu/tree/master/tools/unitrace

### 1.2 Build

**Prerequisites:**
- CMake >= 3.22
- C++17 compiler
- Intel oneAPI Base Toolkit (source `setvars.sh` first)

**Build steps:**

```bash
# Setup oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Clone pti-gpu (if not already present)
git clone https://github.com/intel/pti-gpu.git
cd pti-gpu/tools/unitrace

# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# The binary is at: build/unitrace
```

### 1.3 Common Usage: `-k` (Metric Sampling) and `-q` (Metric Query)

#### `-k` (Time-based metric sampling)

Samples hardware counters at a fixed time interval while kernels run. Good for seeing how metrics vary over time within a kernel.

```bash
# Basic: sample ComputeBasic metrics (default group)
unitrace -k -o perf.csv ./myapp

# Specify metric group
unitrace -k -g ComputeBasic -o perf.csv ./myapp

# Adjust sampling interval (default 50us, smaller = higher resolution)
unitrace -k -i 20 -o perf.csv ./myapp
```

Output files:
- `perf.<pid>.csv` — device timing
- `perf.metrics.<pid>.csv` — sampled metric values
- `myapp.<pid>.json` — timeline (if `--chrome-kernel-logging` used)

#### `-q` (Metric query per kernel instance)

Queries aggregated hardware counters for each kernel invocation. Gives one row of counters per kernel instance.

```bash
# Basic query
unitrace -q -o perfquery.csv ./myapp

# Specify metric group
unitrace -q -g ComputeBasic -o perfquery.csv ./myapp
```

Output: `perfquery.<pid>.csv` — one row per kernel instance with all metrics in the group.

#### When to use `-k` vs `-q`

| | `-k` (sampling) | `-q` (query) |
|---|---|---|
| Granularity | Time-based samples within kernel | One aggregate per kernel instance |
| Overhead | Lower (sampling) | Higher (instrumentation) |
| Use case | See metric trends over time, long kernels | Get precise per-kernel counters |
| Best for | Bandwidth utilization analysis | Quick bottleneck identification |

### 1.4 Available Metric Groups (Intel Arc / BMG)

| Group | Description | Key Metrics |
|---|---|---|
| ComputeBasic | General compute metrics | XVE_ACTIVE, XVE_STALL, GPU_MEMORY_BYTE_READ/WRITE, XVE_THREADS_OCCUPANCY_ALL |
| DeviceCacheProfile | Cache behavior | L3 hits/misses, LSC accesses |
| MemoryProfile | Memory subsystem | L3 read/write, GPU memory read/write bytes |
| VectorEngineProfile | XVE pipe utilization | ALU0/ALU1/SEND/XMX utilization |
| VectorEngineStalls | XVE stall breakdown | Stall reasons (control, pipe, send, dist, sbid, sync, etc.) |
| EuStallSampling | EU stall at IP level | Per-instruction stall events |

**常用 group**: GPGPU 场景主要用 `ComputeBasic`、`MemoryProfile`、`VectorEngineProfile`、`VectorEngineStalls`。

**使用方式**: `-g` 参数接 group 名称，例如 `-g ComputeBasic`。
可通过 `unitrace --metric-list` 查看当前设备上所有可用 group 及其包含的 metrics。

### 1.5 Intel GPU 数据流与背压模型

#### Load 和 Store 路径

以下示意图展示了数据在 Intel GPU 内存层级中的流动路径，以及每一层可观测的 profiler metrics。

```
                            ┌─────────────────────────────────────────────┐
                            │                    XVE                      │
                            │  (Vector Engine, executes SIMD16/32 threads)│
                            │                                             │
                            │  Stalls: XVE_STALL_SBID (wait for send)    │
                            │          XVE_STALL_SENDWR (send port busy) │
                            │          XVE_STALL_ALUWR (ALU writeback)   │
                            └──────┬────────────────────┬─────────────────┘
                                   │                    │
                          load send│                    │store send
                          (request)│                    │(fire-and-forget)
                                   │                    │
                  ┌────────────────▼────────────────────▼──────────────────┐
                  │              LSC (Load/Store Cache = L1)               │
                  │                   256 KB per Xe-core                   │
   XVE→LSC       │                                                        │
   metrics:      │  ┌──────────────────┐    ┌───────────────────────┐     │
                  │  │   Load Path      │    │    Store Path          │     │
   RD_MSG ───────│─▶│                  │    │                       │◀─── WR_MSG
   REG_REQ ──────│─▶│  ┌────────────┐  │    │  Bypass L1 cache      │◀─── REG_REQ
                  │  │  │ rd_tracker │  │    │  (BYTE_WRITE = 0)     │     │
                  │  │  │  (request  │  │    │  Write-through to L3  │     │
                  │  │  │   queue)   │  │    │       │               │     │
                  │  │  └─────┬──────┘  │    │       │               │     │
                  │  │  Coalesce addrs  │    │       │               │     │
                  │  │  Check L1 tags   │    │       │               │     │
                  │  │       │          │    │       │               │     │
   REG_RSP ◀─────│──│  Hit? ├─Yes──┐   │    │       │               │     │
   (load data)   │  │       │      │   │    │       │               │     │
                  │  │       No     │   │    └───────┼───────────────┘     │
                  │  │       │      │   │            │                     │
                  │  │  ┌────▼───┐  │   │            │                     │
                  │  │  │rd_ret_ │  │   │            │                     │
                  │  │  │tracker │◀─┘   │            │                     │
                  │  │  │(return │      │            │                     │
                  │  │  │ queue) │      │            │                     │
                  │  │  └────┬───┘      │            │                     │
                  │  │       │ to XVE   │            │                     │
                  │  └───────┼──────────┘            │                     │
                  │          │                       │                     │
   L1 metrics:   │   BYTE_READ                      │                     │
   ACCESS ───────│── (from L1 hit                    │                     │
   HIT ──────────│──  or L3 return)                  │                     │
   PARTIAL_WR ───│── (always 0,                      │                     │
                  │   stores bypass)                  │                     │
                  │                                   │                     │
   LSC stalls:   │  rd_tracker_full                  │                     │
                  │  rd_return_tracker_full            │                     │
                  └──────────┼────────────────────────┼─────────────────────┘
                             │                        │
                    L1 miss  │                        │ store write-through
                    (read)   │                        │ (always, every store)
                             │                        │
              ┌──────────────▼────────────────────────▼──────────────────────┐
              │                                                              │
   LSC→L3    │    LSC_L3_READ                     LSC_L3_WRITE              │
   metrics:  │    (L1 miss)                       (write-through)           │
              │                                                              │
              │    ┌────────────┐                                            │
              │    │ L3 superQ  │ ← queue full = L3_SUPERQ_FULL             │
              │    └─────┬──────┘                                            │
              │          ▼                                                   │
              ▼                                                              ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                         L3 (Device Cache)                               │
   │                    18 MB, 24 banks, CL=64B                              │
   │                    Write granularity: DW (4B) enables, no byte          │
   │                    enables — sub-DW writes (e.g. d16u32) require        │
   │                    separate L3 transactions per byte position           │
   │                                                                         │
   │  Read hit:  return data to LSC              Write:                      │
   │  Read miss: fetch from DRAM (L3_MISS)        Full CL → merge           │
   │                                              Partial CL → RFO          │
   │                                              (must read CL first)      │
   │                                                                         │
   │  L3_READ, L3_WRITE     (total requests)                                │
   │  L3_HIT, L3_MISS       (hit/miss)                                      │
   │  L3_STALL               (pipeline stall)                               │
   │  L3_SUPERQ_FULL         (input queue full → backpressure to LSC)       │
   │  std.l3.wr.partial     (partial write count)                            │
   │  std.l3.node_bank.*    (bank utilization)                               │
   │  std.l3.lnep_cc.stall  (CC pipeline stall)                             │
   │                                                                         │
   │  Eviction: dirty CL → writeback to DRAM                                │
   │            clean CL → discard (no writeback)                            │
   │            std.l3.node.cc.evict                                         │
   └──────────┬──────────────────────────────────────┬───────────────────────┘
              │                                      │
     L3 miss  │                               dirty eviction
     (read)   │                               (writeback)
              │                                      │
              ▼                                      ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                        DRAM (Global Memory)                             │
   │                Peak BW: 456 GB/s, 12 channels, CL=256B                  │
   │                                                                         │
   │  ┌─────────────────┐                                                    │
   │  │  request queue   │ ← queue full = GPU_MEMORY_REQUEST_QUEUE_FULL      │
   │  └─────────────────┘                                                    │
   │                                                                         │
   │  GPU_MEMORY_BYTE_READ / _WRITE         (total bytes)                   │
   │  GPU_MEMORY_BYTE_READ_RATE / _WRITE_RATE (throughput)                  │
   │  GPU_MEMORY_L3_READ  (read requests from L3 miss)                      │
   │  GPU_MEMORY_L3_WRITE (writeback requests from L3 eviction)             │
   │  GPU_MEMORY_64B_TRANSACTION_READ / _WRITE                              │
   │  COMPRESSOR_INPUT / _OUTPUT (HW compression ratio)                     │
   └──────────────────────────────────────────────────────────────────────────┘
```

#### 背压模型 (Backpressure Model)

Read 路径中，每一层都有有限队列。当下游队列满时，上游被阻塞，stall 逐级回传到 XVE：

```
XVE ──send──▶ [rd_tracker queue] ──miss──▶ [L3 superQ] ──miss──▶ [DRAM queue]
     ◀──────  [rd_return_tracker] ◀──hit/fill── L3     ◀──fill──  DRAM
```

**各级队列满信号（对应 stall metrics）：**

| 队列满信号 | 含义 |
|-----------|------|
| `rd_tracker_full` | LSC 无法接受 XVE 新的 read 请求 |
| `rd_return_tracker_full` | LSC return buffer 满，L3 数据无法送达 |
| `L3_SUPERQ_FULL` | L3 输入队列满，LSC 请求被阻塞 |
| `L3_STALL` | L3 管线 stall（bank conflict 等） |
| `GPU_MEMORY_REQUEST_QUEUE_FULL` | DRAM 请求队列满 |

当任意队列满时，整条链路向上 stall 直到 XVE（表现为 `SF_HOLD` = Shared Function access hold，或 `XVE_STALL_SBID` 等待 load 数据返回）。

#### Load Path (Step by Step)

1. **XVE 发出 load send**（如 `load.ugm.d32.a64`），携带 32 (SIMD32) 或 16 (SIMD16) 个 lane 地址
   - Metrics: `XVE_LOAD_STORE_CACHE_READ_MESSAGE_COUNT` (+1/send), `XVE_LOAD_STORE_CACHE_REGISTER_REQUEST_COUNT` (+REQ_per_send)
   - XVE 分配 SBID (Scoreboard ID)，在数据返回前 stall 在 `XVE_STALL_SBID`

2. **LSC 地址合并**: 将 32 个地址按 cache line (64B) 合并，连续 lane 访问同一 CL 会被 merge
   - Metrics: `LOAD_STORE_CACHE_ACCESS` (+1/CL touched), `LOAD_STORE_CACHE_HIT` (+1/CL hit in L1)
   - 示例: d32 SIMD32 连续 → 2 CLs; d32 SIMD32 stride → 最多 32 CLs; d32x4 SIMD32 → 8 CLs

3. **L1 命中**: 数据直接从 L1 返回 XVE
   - Metrics: `LOAD_STORE_CACHE_BYTE_READ` (+bytes returned), `XVE_LOAD_STORE_CACHE_REGISTER_RESPONSE_COUNT` (+RSP_per_send)
   - `std.lsc.msg.latency` 记录平均往返时间

4. **L1 未命中**: 请求转发到 L3
   - Metrics: `LOAD_STORE_CACHE_L3_READ` (+1/missed CL)
   - **L3 命中**: 数据返回 LSC → L1 缓存 (clean) → 返回 XVE
   - **L3 未命中**: `L3_MISS` +1, 请求发往 DRAM (`GPU_MEMORY_L3_READ` +1, `GPU_MEMORY_BYTE_READ` +64B)
   - DRAM 返回数据 → L3 缓存 → LSC → L1 → XVE

5. **XVE 收到数据**, 清除 SBID, 恢复执行

#### Store Path (Step by Step)

1. **XVE 发出 store send**（如 `store.ugm.d32.a64`）
   - Metrics: `XVE_LOAD_STORE_CACHE_WRITE_MESSAGE_COUNT` (+1), `REG_REQ` (+REQ_per_send)
   - Store 对 XVE 来说是 fire-and-forget，不等待数据返回（但 SBID 用于 send port 排序）

2. **LSC 完全跳过 L1 cache** — store 数据不写入 L1
   - `LOAD_STORE_CACHE_BYTE_WRITE` = 0（在 B580 上所有 kernel 均确认）
   - `LOAD_STORE_CACHE_PARTIAL_WRITE_COUNT` = 0（L1 从不接收 store 数据）
   - `LOAD_STORE_CACHE_ACCESS` 仍会递增（LSC 处理地址，但数据直通 L3）

3. **Write-through 到 L3**: LSC 生成 L3 写请求
   - Metrics: `LOAD_STORE_CACHE_L3_WRITE` (+CLs), `L3_WRITE` (+CLs)
   - **d16u32 写放大**: 单条 SIMD32 d16u32 store (stride=2B) 产生 4× L3 write 请求（even/odd byte-lane 拆分，因 L3 只有 DW enables 无 byte enables）。d32 store 则产生预期数量的 CL writes

4. **L3 处理写入**:
   - **Full CL write** (d32 连续): L3 合并数据，标记 CL dirty，无需读 DRAM
   - **Partial CL write** (d16u32 或交错 d32): L3 必须先读取已有 CL（Read-For-Ownership / RFO），再合并 partial 数据
     - RFO 产生额外 DRAM 读: `GPU_MEMORY_BYTE_READ` (+64B), `GPU_MEMORY_L3_READ` (+1) — 这些是 store 引起的额外 DRAM 读
     - `std.l3.wr.partial` 记录 partial write 次数

5. **L3 驱逐**: L3 满时，dirty CL 写回 DRAM
   - Metrics: `std.l3.node.cc.evict`, `GPU_MEMORY_L3_WRITE` (+1/dirty CL evicted), `GPU_MEMORY_BYTE_WRITE` (+64B)

## Part 2: Nsight Compute (NVIDIA CUDA)

### 2.1 What is Nsight Compute (ncu)

Nsight Compute 是 NVIDIA 的 CUDA kernel 级性能分析工具，可以：
- 收集每个 kernel 的硬件性能计数器（SM utilization、memory throughput、occupancy 等）
- 通过 sections 机制组织指标，提供 roofline、memory chart、warp stall 等分析视角
- 支持 kernel replay 以收集完整的多 pass 计数器数据
- 提供 CLI (`ncu`) 和 GUI (`ncu-ui`) 两种使用方式
- 支持源码级别 (SASS/PTX) 的 metric 关联

官方文档: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html

### 2.2 安装与环境

ncu 随 CUDA Toolkit 一起安装，通常位于：
```bash
/usr/local/cuda/bin/ncu
/usr/local/cuda/bin/ncu-ui
```

确认版本：
```bash
ncu --version
```

**注意事项：**
- 需要 root 权限或设置 `CAP_SYS_ADMIN`，否则只能 profile 自己启动的进程
- 非 root 用户可通过以下方式临时开启：
  ```bash
  sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
  # 或
  echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
  ```

### 2.3 常用命令行用法

#### 基本 profile（收集默认 basic set）

```bash
# Profile 所有 kernel，结果保存到文件
ncu -o profile ./myapp

# 结果保存为 profile.ncu-rep，可用 ncu-ui 打开
```

#### 指定 kernel 过滤

```bash
# 按 kernel 名称过滤（精确匹配）
ncu -k softmax_kernel -o profile ./myapp

# 按 kernel 名称正则匹配
ncu -k regex:softmax -o profile ./myapp

# 只 profile 前 N 个 kernel launch
ncu -c 5 -o profile ./myapp

# 跳过前 N 个 kernel，再 profile M 个
ncu -s 10 -c 5 -o profile ./myapp
```

#### 指定 Section 收集

```bash
# 查看所有可用 sections
ncu --list-sections

# 查看所有可用 sets (section 组合)
ncu --list-sets

# 收集特定 section
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis -o profile ./myapp

# 收集完整 set（full = 所有 section）
ncu --set full -o profile ./myapp

# 收集 detailed set（比 basic 更多，比 full 更少）
ncu --set detailed -o profile ./myapp
```

#### 收集特定 metric

```bash
# 查看所有可用 metrics
ncu --query-metrics

# 收集指定 metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed -o profile ./myapp
```

#### 直接在终端打印结果（不保存文件）

```bash
# 默认 details page
ncu -k regex:softmax ./myapp

# CSV 格式输出，方便后处理
ncu --csv -k regex:softmax ./myapp

# 显示 raw page（所有原始 metrics）
ncu --page raw -k regex:softmax ./myapp
```

### 2.4 Profile PyTorch 程序

对于 PyTorch 程序，kernel 名称通常较长且复杂，推荐用正则匹配：

```bash
# Profile softmax 相关 kernel
ncu -k regex:softmax -c 3 --set detailed -o softmax_profile python my_softmax_test.py

# Profile 所有 kernel，但只取前几个
ncu -c 10 --set basic -o profile python train.py

# 使用 NVTX 过滤特定区域（需在 PyTorch 代码中用 torch.cuda.nvtx.range_push/pop）
ncu --nvtx --nvtx-include "softmax/" -o profile python my_test.py
```

**PyTorch 脚本示例（用于 ncu profile）：**
```python
import torch

def test_softmax():
    x = torch.randn(1024, 1024, device='cuda')
    # warmup
    for _ in range(10):
        y = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    # 实际 profile 的迭代
    y = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

if __name__ == "__main__":
    test_softmax()
```

```bash
# 跳过 warmup kernel，只 profile 最后一次
ncu -k regex:softmax -s 10 -c 1 --set detailed -o softmax_profile python test_softmax.py
```

### 2.5 常用 Sections 说明

| Section | 标识符 | 分析内容 |
|---|---|---|
| GPU Speed Of Light Throughput | `SpeedOfLight` | SM 和 Memory 的利用率百分比，快速判断 compute-bound 还是 memory-bound |
| Memory Workload Analysis | `MemoryWorkloadAnalysis` | 各级 memory (L1/L2/DRAM) 的吞吐、命中率 |
| Compute Workload Analysis | `ComputeWorkloadAnalysis` | SM pipe 利用率 (FMA/ALU/Tensor Core 等) |
| Occupancy | `Occupancy` | Achieved vs theoretical occupancy，限制因素 |
| Launch Statistics | `LaunchStats` | Grid/Block 维度、寄存器数、shared memory 用量 |
| Warp State Statistics | `WarpStateStatistics` | Warp stall 原因分布 (barrier, memory, math pipeline 等) |
| Instruction Statistics | `InstructionStatistics` | 指令执行统计 |
| Source Counters | `SourceCounters` | 源码级别 (SASS) 热点指标 |
| Roofline | `SpeedOfLight_RooflineChart` | Roofline 模型图（Arithmetic Intensity vs Performance） |

**推荐组合：**
- 快速定位瓶颈：`--set basic`（包含 SpeedOfLight + LaunchStats + Occupancy）
- 深入分析：`--set detailed`（增加 MemoryWorkloadAnalysis + WarpStateStatistics 等）
- 完整分析：`--set full`（所有 sections，overhead 最大）

### 2.6 关键 Metrics 速查

| Metric | 含义 |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 计算吞吐利用率 (%) |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 带宽利用率 (%) |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1/Tex cache 吞吐利用率 |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2 cache 吞吐利用率 |
| `sm__warps_active.avg.pct_of_peak_sustained_elapsed` | Achieved occupancy |
| `smsp__average_warp_latency_per_inst_issued.ratio` | 平均 warp 延迟（stall cycles per issued instruction） |
| `dram__bytes_read.sum` | DRAM 总读取字节数 |
| `dram__bytes_write.sum` | DRAM 总写入字节数 |
| `sm__sass_thread_inst_executed_op_fadd_pred_on.sum` | FP32 加法指令数 |
| `sm__sass_thread_inst_executed_op_fmul_pred_on.sum` | FP32 乘法指令数 |
| `sm__sass_thread_inst_executed_op_ffma_pred_on.sum` | FP32 FMA 指令数 |

### 2.7 结果分析工作流

1. **第一步：Speed Of Light** — 看 SM% 和 Memory%，判断 compute-bound 还是 memory-bound
   - SM% 高 + Memory% 低 → compute-bound
   - SM% 低 + Memory% 高 → memory-bound
   - 两者都低 → latency-bound（通常是 occupancy 或 stall 问题）

2. **第二步：根据瓶颈深入**
   - Memory-bound → 看 MemoryWorkloadAnalysis，检查 L1/L2 命中率、DRAM 带宽
   - Compute-bound → 看 ComputeWorkloadAnalysis，检查哪个 pipe 是瓶颈
   - Latency-bound → 看 WarpStateStatistics + Occupancy

3. **第三步：Warp Stall 分析** — 看 stall 原因分布
   - `stall_barrier` → sync 过多
   - `stall_long_scoreboard` → 等待全局内存
   - `stall_short_scoreboard` → 等待 shared memory 或 L1
   - `stall_math_pipe_throttle` → 计算单元饱和
   - `stall_mio_throttle` → memory I/O 队列满

4. **第四步：源码关联** — 用 `--set full` 或 `--section SourceCounters` 查看热点源码行

### 2.8 `-k` vs `-q` 对比 (与 Unitrace 类比)

| | NCU (ncu) | Unitrace `-q` |
|---|---|---|
| 粒度 | 每个 kernel launch 的完整 metrics | 每个 kernel launch 的聚合 metrics |
| 数据收集方式 | Kernel replay (多 pass) | 硬件计数器查询 |
| Overhead | 较高（replay） | 中等 |
| 输出 | `.ncu-rep` 文件或终端 | CSV 文件 |
| 可视化 | ncu-ui GUI | 需手动处理 |
| 适用场景 | 深度 kernel 优化 | 快速性能概览 |

### 2.9 实用技巧

```bash
# 1. 对比两次 profile（A/B test）
ncu-ui baseline.ncu-rep  # 在 GUI 中 Add Baseline

# 2. 从已有 report 中导出特定 kernel
ncu --import old_report.ncu-rep --export filtered.ncu-rep -k regex:softmax

# 3. 强制覆盖已有输出文件
ncu -f -o profile ./myapp

# 4. 多 GPU 场景只 profile 指定 GPU
ncu --devices 0 -o profile ./myapp

# 5. Profile 自定义 CUDA kernel（指定源码路径以启用源码关联）
ncu --set full --import-source yes -o profile ./myapp

# 6. 查看某个 metric 的所有可用后缀
ncu --query-metrics-mode suffix --metrics sm__throughput
```

