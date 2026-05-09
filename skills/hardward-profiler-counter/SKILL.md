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

#### GPGPU Profiling 关注的关键 Counters

以下按分析维度整理 GPGPU 场景中最常关注的 counters。实际 profiling 时，建议先跑 `ComputeBasic` 做全局概览，再根据瓶颈方向选择对应 group 深入。

##### 总览 & 时间 (ComputeBasic)

| Counter | 含义 | 关注原因 |
|---|---|---|
| `GpuTime[ns]` | Kernel 在 GPU 上的执行时间 | 最基本的性能指标，用于对比优化前后 |
| `GpuCoreClocks[cycles]` | GPU 核心时钟周期数 | 配合 GpuTime 可验证频率是否稳定 |
| `AvgGpuCoreFrequencyMHz[MHz]` | 平均核心频率 | 频率波动会影响性能对比的公平性，需确保两次 run 频率一致 |

##### XVE 利用率 & Stall (ComputeBasic / VectorEngineStalls)

| Counter | 含义 | 关注原因 |
|---|---|---|
| `XVE_ACTIVE[%]` | XVE 至少有一条 pipe 在执行的时间比例 | 核心利用率指标，越高越好 |
| `XVE_STALL[%]` | XVE 有线程但无 pipe 活跃的时间比例 | 高 stall 说明有瓶颈，需进一步看 stall 原因 |
| `XVE_SHARED_FUNCTION_ACCESS_HOLD[%]` | XVE 被 Shared Function (LSC/SLM/Sampler) 反压的时间比例 | 高值说明 memory 子系统是瓶颈，XVE send 请求被阻塞 |
| `XVE_THREADS_OCCUPANCY_ALL[%]` | 线程槽位占用率 | 低 occupancy 可能导致无法隐藏 memory latency |
| `XVE_MULTIPLE_PIPE_ACTIVE[%]` | 至少两条 ALU pipe 同时活跃的时间比例 | 反映指令级并行度 (ILP) |

##### 指令执行 (ComputeBasic / VectorEngineProfile)

| Counter | 含义 | 关注原因 |
|---|---|---|
| `XVE_INST_EXECUTED_ALU0_ALL` | ALU0 pipe 执行的指令槽数 | 主要整数/浮点运算 pipe |
| `XVE_INST_EXECUTED_ALU1_ALL` | ALU1 pipe 执行的指令槽数 | 辅助运算 pipe (不含 extended math) |
| `XVE_INST_EXECUTED_ALU2_ALL` | ALU2 pipe (XMX) 执行的指令槽数 | 矩阵运算 pipe，GEMM 相关 kernel 应关注 |
| `XVE_INST_EXECUTED_SEND_ALL` | SEND pipe 执行的指令数 | 所有 load/store/SLM/fence 都走 SEND pipe，高值说明 memory 指令多 |
| `XVE_INST_ISSUED_ALL` | 总解码发射指令数 | 与 EXECUTED 对比可看 issue 效率 |
| `XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION[%]` | ALU0 时间占比 | 快速判断 compute 利用率 |
| `XVE_INST_EXECUTED_ALU1_ALL_UTILIZATION[%]` | ALU1 时间占比 | 同上 |
| `XVE_INST_EXECUTED_ALU2_ALL_UTILIZATION[%]` | ALU2 时间占比 | 同上 |

##### 取指 (ComputeBasic)

| Counter | 含义 | 关注原因 |
|---|---|---|
| `ICACHE_HIT` | 指令缓存命中次数 | 命中率 = HIT/(HIT+MISS)。大 kernel 或代码膨胀导致 ICACHE 压力 |
| `ICACHE_MISS` | 指令缓存未命中次数 | 高 miss 说明 kernel 指令 footprint 超出 ICACHE 容量，取指成为瓶颈。可能的原因：kernel 代码体积过大、分支导致 ICACHE thrashing、过多 inline 展开 |

##### LSC / L1 Cache (ComputeBasic / MemoryProfile)

| Counter | 含义 | 关注原因 |
|---|---|---|
| `LOAD_STORE_CACHE_ACCESS` | LSC 总访问次数 (含 load + store) | 反映 XVE 发出的 memory 请求密度 |
| `LOAD_STORE_CACHE_HIT` | LSC 命中次数 | 命中率 = HIT/ACCESS，高命中率说明数据局部性好 |
| `LOAD_STORE_CACHE_BYTE_READ[bytes]` | LSC 读出的字节数 (不含 SLM) | 实际从 L1 读取的数据量 |
| `LOAD_STORE_CACHE_BYTE_WRITE[bytes]` | LSC 写入的字节数 (不含 SLM) | 注意：某些架构/配置下 store 直通 L3，此值可能为 0 |
| `LOAD_STORE_CACHE_PARTIAL_WRITE_COUNT` | 不完整 subsector 写入次数 | 高值说明存在 partial write 问题 (如 d16u32 store) |

##### LSC 详细 (MemoryProfile)

| Counter | 含义 | 关注原因 |
|---|---|---|
| `XVE_LOAD_STORE_CACHE_READ_MESSAGE_COUNT` | XVE → LSC 的 read message 数 | 一次 send 产生一个 message |
| `XVE_LOAD_STORE_CACHE_WRITE_MESSAGE_COUNT` | XVE → LSC 的 write message 数 | 对比优化前后可验证 store 合并效果 |
| `XVE_LOAD_STORE_CACHE_REGISTER_REQUEST_COUNT` | XVE → LSC 的 register request 数 | 每个 send 可能产生多个 request (向量化宽度) |
| `XVE_LOAD_STORE_CACHE_REGISTER_RESPONSE_COUNT` | LSC → XVE 的 register response 数 | 应与预期读取数据量一致 |
| `LOAD_STORE_CACHE_INPUT_AVAILABLE[%]` | LSC 输入端有请求等待的时间比例 | 高值说明 LSC 持续有压力 |
| `LOAD_STORE_CACHE_OUTPUT_READY[%]` | LSC 输出端有数据就绪的时间比例 | 低值可能说明 L3 返回慢 |
| `LOAD_STORE_CACHE_NUMBER_OF_BANK_ACCESS_COUNT` | LSC bank 访问次数 | 显著高于 ACCESS 说明存在 bank conflict |

##### SLM (ComputeBasic / MemoryProfile)

| Counter | 含义 | 关注原因 |
|---|---|---|
| `SLM_BYTE_READ[bytes]` | SLM 读取字节数 | 使用 SLM 做 reduction/共享的 kernel 关注 |
| `SLM_BYTE_WRITE[bytes]` | SLM 写入字节数 | 同上 |
| `SLM_BANK_CONFLICT_COUNT` | SLM bank conflict 次数 | 高值说明 SLM 访问模式有 bank conflict，影响性能 |
| `SLM_ACCESS_COUNT` | SLM 总访问次数 | 配合 bank conflict 计算 conflict 率 |
| `XVE_SLM_READ_MESSAGE_COUNT` | XVE 发出的 SLM read message 数 | 反映 SLM 读取的 message 粒度 |
| `XVE_SLM_WRITE_MESSAGE_COUNT` | XVE 发出的 SLM write message 数 | 反映 SLM 写入的 message 粒度 |
| `XVE_SLM_ATOMIC_MESSAGE_COUNT` | XVE 发出的 SLM atomic 操作数 | SLM atomic 会序列化，高值可能成为瓶颈 |
| `XVE_SLM_FENCE_MESSAGE_COUNT` | XVE 发出的 SLM fence 操作数 | fence 用于同步，过多会增加延迟 |

##### L3 / Device Cache (ComputeBasic / DeviceCacheProfile)

| Counter | 含义 | 关注原因 |
|---|---|---|
| `L3_READ` | L3 64B 读请求数 | 来自 LSC miss 的读请求 |
| `L3_WRITE` | L3 64B 写请求数 | 来自 LSC 的写请求。**关键**: d16u32 store 会导致 4x 写放大 |
| `L3_HIT` | L3 命中次数 | 命中率 = HIT/(HIT+MISS) |
| `L3_MISS` | L3 未命中次数 | miss 触发 DRAM 读取 |
| `L3_STALL[%]` | L3 bank stall 时间比例 | 高值说明 L3 是瓶颈 (bank conflict / 带宽饱和) |
| `L3_BUSY[%]` | L3 繁忙时间比例 | 接近 100% 说明 L3 持续有工作 |
| `L3_SUPERQ_FULL[%]` | L3 输入队列满的时间比例 | **关键背压指标**: 高值说明 L3 来不及处理请求，LSC 被阻塞 |
| `L3_INPUT_AVAILABLE[%]` | L3 输入端有请求的时间比例 | 高值说明 L3 持续承受高压力 |
| `L3_OUTPUT_READY[%]` | L3 输出端有数据就绪的时间比例 | 反映 L3 产出数据的频率 |
| `L3_ATOMIC_ACCESS` | L3 原子操作次数 | atomic 操作会序列化，高值可能是瓶颈 |

##### L3 详细 (DeviceCacheProfile)

| Counter | 含义 | 关注原因 |
|---|---|---|
| `LOAD_STORE_CACHE_L3_READ` | LSC → L3 的读请求 (L1 miss) | 即 LSC miss 导致的 L3 读 |
| `LOAD_STORE_CACHE_L3_WRITE` | LSC → L3 的写请求 | 即 LSC store 产生的 L3 写 |
| `LOAD_STORE_CACHE_L3_HIT` | LSC → L3 请求中 L3 命中的次数 | 配合 L3_READ 计算 LSC 请求的 L3 命中率 |
| `ICACHE_L3_READ` | 指令缓存 → L3 的读请求 | ICACHE miss 后去 L3 取指 |
| `ICACHE_L3_HIT` | 指令缓存 → L3 的读请求中命中的次数 | 若 ICACHE miss 且 L3 也 miss，取指延迟极高 |

##### DRAM / Global Memory (ComputeBasic / MemoryProfile)

| Counter | 含义 | 关注原因 |
|---|---|---|
| `GPU_MEMORY_BYTE_READ[bytes]` | DRAM 读取总字节数 | 与理论 IO 对比可检测读放大 (如 RFO) |
| `GPU_MEMORY_BYTE_WRITE[bytes]` | DRAM 写入总字节数 | 与理论 IO 对比可检测写放大 |
| `GPU_MEMORY_BYTE_READ_RATE[GBpS]` | DRAM 读带宽 | 与峰值带宽对比计算利用率 |
| `GPU_MEMORY_BYTE_WRITE_RATE[GBpS]` | DRAM 写带宽 | 同上 |
| `GPU_MEMORY_L3_READ` | L3 miss 触发的 DRAM 读请求数 | L3 miss → DRAM read |
| `GPU_MEMORY_L3_WRITE` | L3 脏行驱逐导致的 DRAM 写请求数 | dirty eviction → DRAM write |
| `GPU_MEMORY_64B_TRANSACTION_READ` | DRAM 64B 读事务数 | 精确的 DRAM 事务计数 |
| `GPU_MEMORY_64B_TRANSACTION_WRITE` | DRAM 64B 写事务数 | 同上 |
| `GPU_MEMORY_REQUEST_QUEUE_FULL[%]` | DRAM 请求队列满的时间比例 | **背压指标**: 高值说明 DRAM 带宽饱和 |
| `TLB_MISS` | TLB 未命中次数 | 高值可能说明数据访问跨越太多 page，或 working set 太大 |

##### 压缩 (ComputeBasic)

| Counter | 含义 | 关注原因 |
|---|---|---|
| `COMPRESSOR_INPUT` | 压缩器输入的 256B 写次数 | 压缩率 = OUTPUT/INPUT |
| `COMPRESSOR_OUTPUT` | 压缩器输出的 256B 写次数 | 低于 INPUT 说明硬件压缩有效，节省 DRAM 带宽 |

#### Profiling 分析思路

1. **先看 XVE_ACTIVE vs XVE_STALL** — 判断 kernel 是 compute-bound 还是 stall-bound
2. **若 stall 高，看 XVE_SHARED_FUNCTION_ACCESS_HOLD** — 是否被 memory 子系统反压
3. **看 L3_STALL 和 L3_SUPERQ_FULL** — 判断 L3 是否拥塞
4. **看 GPU_MEMORY_BYTE_READ/WRITE vs 理论值** — 检测读/写放大
5. **看 ICACHE_HIT/MISS** — 大 kernel 可能有取指瓶颈
6. **看 SLM_BANK_CONFLICT_COUNT** — 使用 SLM 的 kernel 检查 bank conflict

**常用 group**: GPGPU 场景主要用 `ComputeBasic`、`MemoryProfile`、`DeviceCacheProfile`、`VectorEngineProfile`、`VectorEngineStalls`。

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
   XVE→LSC        │                                                        │
   metrics:       │  ┌──────────────────┐    ┌───────────────────────┐     │
                   │  │   Load Path      │    │    Store Path          │     │
   XVE→LSC        │  │                  │    │                       │◀─── WR_MSG
   metrics:       │  │  ┌────────────┐  │    │  Store 经 LSC 转发    │◀─── REG_REQ
                   │  │  │ rd_tracker │  │    │  到 L3                │     │
                   │  │  │  (request  │  │    │                       │     │
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
   PARTIAL_WR ───│──                                   │                     │
                  │                                   │                     │
   LSC stalls:   │  rd_tracker_full                  │                     │
                  │  rd_return_tracker_full            │                     │
                  └──────────┼────────────────────────┼─────────────────────┘
                             │                        │
                     L1 miss  │                        │ store
                     (read)   │                        │
                             │                        │
              ┌──────────────▼────────────────────────▼──────────────────────┐
              │                                                              │
   LSC→L3    │    LSC_L3_READ                     LSC_L3_WRITE              │
    metrics:  │    (L1 miss)                       (store)               │
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

2. **LSC 将 store 请求发往 L3** — 具体的 L1 缓存策略取决于架构和 cache control hint
   - `LOAD_STORE_CACHE_BYTE_WRITE`（在 B580 上观察到为 0，但不确定是否所有架构/配置均如此）
   - `LOAD_STORE_CACHE_PARTIAL_WRITE_COUNT`
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

## Part 2: Nsight Compute
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

