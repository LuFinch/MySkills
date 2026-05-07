# GPU Memory Hierarchy Skill

GPU 内存层级知识库，重点覆盖 Intel GPU (Xe 架构) 的内存体系。

> Compatible with: Claude Code, OpenCode, GitHub Copilot, Roo Code, and other AI coding assistants.

## Overview

本 skill 提供 GPU 内存层级的参考知识，帮助 agent 在性能分析、kernel 优化、内存瓶颈诊断等场景中做出准确判断。

## GPU 通用内存层级

从快到慢、从小到大：

```
Registers (寄存器)
  ↓
Shared Local Memory / SLM (共享本地内存)
  ↓
L1 Cache (Load/Store Cache)
  ↓
L3 Cache (Last Level Cache, LLC)
  ↓
Global Memory / VRAM (HBM / GDDR)
  ↓
Host Memory (CPU DRAM, via PCIe/CXL)
```

| 层级 | 典型延迟 | 典型带宽 | 容量范围 | 对应 SYCL/oneAPI 概念 |
|------|---------|---------|---------|---------------------|
| Registers | ~1 cycle | N/A (内部) | 每线程几十~几百 bytes | private memory |
| SLM | ~20-30 cycles | 数 TB/s (聚合) | 64-128 KB/work-group | local memory |
| L1 Cache | ~20-50 cycles | 数 TB/s | 数十~数百 KB/slice | 隐式 (硬件管理) |
| L3 Cache (LLC) | ~100-200 cycles | 数百 GB/s ~ 数 TB/s | 数 MB ~ 数百 MB | 隐式 (硬件管理) |
| Global Memory | ~300-500 cycles | 数十~数千 GB/s | 数 GB ~ 数百 GB | global memory |

> 注意：以上延迟和带宽为数量级参考，实际数值因架构和频率而异。

## Intel GPU 架构内存层级

### Xe-HPC (Ponte Vecchio / Data Center GPU Max)

```
GRF (General Register File)
  512 bytes/thread (128 × 32-bit registers), 8 threads/EU
  ↓
SLM + L1 Cache (共享同一块 SRAM)
  128 KB per Xe-core, SLM 软件管理 / L1 硬件管理
  ↓
L3 Cache (Last Level Cache, LLC, 全 GPU 共享)
  ~384 MB (全芯片, PVC Max 1550 双 tile 合计)
  ↓
HBM2e (Global Memory)
  128 GB (Max 1550), 峰值带宽 ~3.27 TB/s (双 tile 合计)
```

**关键参数 (PVC Max 1550, 双 tile):**
- EU 数量: 1024 (512/tile)
- Xe-cores: 128 (64/tile)
- SLM: 128 KB × 128 = 16 MB 总计
- L3 Cache (LLC): ~384 MB
- HBM2e: 128 GB, 8 stacks × 2 tiles
- 峰值内存带宽: ~3.27 TB/s

### Xe2-HPG (Battlemage / Arc B 系列) — 示例: Intel Arc B580

```
GRF (General Register File)
  512 bytes/thread (128 × 32-bit registers), 8 threads/XVE (EU)
  ↓
SLM + L1 Cache (共享同一块 SRAM)
  256 KB per Xe-core (不使用 SLM 时全部可用作 L1; 默认 bspec 划分 160 KB SLM + 96 KB L1)
  Read: 256 B/clk per Xe-core, Write: 128 B/clk per Xe-core (32 B/clk per XVE)
  Cache line size: 64 B
  ↓
L3 Cache (Last Level Cache, LLC, 全 GPU 共享)
  18 MB total (6 nodes × 4 banks/node = 24 banks, 768 KB/bank)
  64 B/clk per bank, CL size = 64 B
  ↓
Hardware Compressor (CCS)
  位于 L3 和 DRAM 之间, 压缩粒度 64 B, 压缩缓存 32 KB
  ↓
GDDR6 (Global Memory)
  256-bit bus, 峰值带宽 456 GB/s
```

**Intel Arc B580 内存层级示意图:**

```
                    ┌─────────────────────────────────┐
                    │           XVE (EU)              │
                    │  GRF: 128 regs × 64B = 8 KB     │
                    └──────────┬──────────┬───────────┘
                         load  │          │ store
                      256 B/clk│ 128 B/clk│  (per Xe-core)
                    ┌──────────▼──────────▼────────────┐
                    │     L1 (LSC) — 256 KB/Xe-core    │
                    │     CL = 64B                     │
                    │     Stores: write-through (bypass│
                    │     L1 data, go directly to L3)  │
                    └──────────┬──────────┬────────────┘
                      L1 miss  │          │ write-through
                      64 B/clk │  per bank│
                    ┌──────────▼──────────▼────────────┐
                    │   L3 — 18 MB (24 banks × 768 KB) │
                    │   CL = 64B (verified)            │
                    └──────────┬──────────┬────────────┘
                      L3 miss  │          │ evict/writeback
                      32 B/clk │  per CMI │
                    ┌──────────▼──────────▼────────────┐
                    │     DRAM — GDDR6, 456 GB/s       │
                    │     CL = 256B                    │
                    └──────────────────────────────────┘
```

**关键参数 (Intel Arc B580, BMG-G21 X2):**
- 架构: 5 slices × 4 Xe-cores/slice = 20 Xe-cores
- XVE (EU) 数量: 8 XVEs/Xe-core × 20 = 160 XVEs
- 线程: 8 threads/XVE
- Boost 频率: 2.85 GHz
- L1 (LSC): 256 KB/Xe-core, CL = 64 B
- L3 (LLC): 18 MB, 24 banks, CL = 64 B
- GDDR6: 256-bit bus, 峰值带宽 456 GB/s
- 硬件压缩: L3-DRAM 之间 CCS 压缩器

## 关键概念

### GRF (General Register File)

- Intel GPU 每个 EU 的寄存器文件
- Xe-HPC/HPG: 每线程 128 个 32-bit 寄存器 (512 bytes)
- 大寄存器模式 (Large GRF): 每线程 GRF 翻倍至 256 × 32-bit (1 KB/thread)，但每 EU 的并发线程数减半 (8→4)，每 EU 的总 GRF 容量不变
- 寄存器压力过大会导致 spill 到内存，严重影响性能

### SLM (Shared Local Memory) & L1 Cache

- SLM 和 L1 Cache 共享同一块 SRAM，位于每个 Xe-core 内部
- 硬件根据 kernel 的 SLM 分配需求动态划分这块 SRAM
- SLM 部分由软件显式管理，L1 部分由硬件自动管理
- SLM 对应 CUDA 的 shared memory
- 在同一 work-group 内的 work-items 之间共享
- 用途：数据复用、work-item 间通信、reduction
- Xe-HPC: 128 KB/Xe-core (SLM + L1 共享)
- Xe-HPG: 64 KB/Xe-core (SLM + L1 共享)
- 带宽极高 (接近寄存器级别)，但容量有限
- SYCL 中通过 `sycl::local_accessor` 使用
- 注意：SLM 分配越多，L1 cache 可用空间越小，反之亦然

### L1 Cache (Load/Store Cache, LSC)

- Intel GPU 的 L1 cache 也称 Load/Store Cache (LSC)，位于每个 Xe-core 内部
- 与 SLM 共享同一块 SRAM（见上节）
- 硬件自动管理，对软件透明
- 用于缓存 global memory 的读写访问
- Intel GPU 没有 L2 cache，L1 之上直接是 L3
- Store 行为: write-through，数据跳过 L1 直接写入 L3

#### LSC Descriptor (Send 指令描述符)

XVE (EU) 通过 send 指令向 LSC 发起内存读写请求，descriptor 定义了每条 send 每个 SIMD lane 搬运的数据量：

| Descriptor | 含义 | Bytes/lane | × 32 lanes (SIMD32) |
|-----------|------|-----------|---------------------|
| d16u32 | 16-bit 数据装在 32-bit 容器中 (零扩展) | 4B (GRF 侧) / 2B (内存侧) | 128B (GRF) / 64B (内存) |
| d32 | 32-bit 数据 | 4B | 128B |
| d32x2 | 2 × 32-bit (向量化) | 8B | 256B |
| d32x4 | 4 × 32-bit (向量化) | 16B | 512B |

**关键细节:**
- **d32x2/d32x4** 是向量化 send，单条指令搬运更多数据，减少指令开销
- **GRF 与数据类型的关系**: GRF 的最小粒度是 32-bit（一个 DW）。fp16 等 16-bit 数据在 GRF 中每个 lane 占 4B（低 16-bit 存数据，高 16-bit 补零），这是硬件寄存器宽度决定的，不需要软件做额外 pack。因此 d16u32 SIMD32 在 GRF 中占 128B，但有效数据仅 64B。若想节省 GRF 空间，可用 d16 descriptor 将两个 fp16 pack 进一个 32-bit 槽位
- **d16u32 store 的 4× 写放大问题**:
  1. 理论上 32 个 fp16 = 64B，刚好一个 CL 即可装下
  2. 但 store 走 write-through 跳过 L1 直接到 L3。LSC 收到 128B 的 send 数据（32 lanes × 4B），知道每个 lane 只有低 16-bit 有效，需要将有效数据写到 L3。然而 L3 写端口只有 DW enables（最小写粒度 4B），没有 byte enables，无法做到"只写一个 DW 的低 2 字节"
  3. 硬件不得不将 even bytes 和 odd bytes 分开发送，导致一条本该 1 个 CL 的写操作被拆成 4 个 L3 write 请求
  4. d32 写 fp32 则不存在此问题，因为每个 lane 写满一个 DW，与 L3 的 DW enables 完全对齐
  5. Read 路径不受影响：读操作以整个 CL (64B) 为粒度从 L3/DRAM 读出，不需要精细的字节写使能控制，读完后由 LSC/GRF 侧提取所需的 16-bit 数据即可（L3_READ 在 h1 和 f1 场景下相同）
- 编译器可能会将相邻的小 descriptor send 合并为更大的向量 send（如两条 d32 合并为一条 d32x2），这在分析 profiler counter 时需注意区分

### L3 Cache (Last Level Cache, LLC)

- Intel GPU 的最后一级缓存，也称 Last Level Cache (LLC)
- 整个 GPU 共享的全局缓存，所有 Xe-core 共用
- 硬件自动管理，对软件透明
- Intel GPU 的 L3 cache 通常较大 (PVC: ~384 MB)
- 对于 data reuse 高的 workload (如 GEMM)，L3 命中率很关键
- 部分 Intel GPU 支持 cache hint (LSC cache control) 来控制数据在 L1/L3 的缓存行为

### Intel GPU Load/Store 数据流

#### Load Path

1. **XVE 发出 load send**（如 `load.ugm.d32.a64`），携带 32 (SIMD32) 或 16 (SIMD16) 个 lane 地址
   - XVE 分配 SBID (Scoreboard ID)，在数据返回前 stall（等待 load 完成）

2. **LSC 地址合并**: 将 32 个地址按 cache line (64B) 合并，连续 lane 访问同一 CL 会被 merge
   - 示例: d32 SIMD32 连续 → 2 CLs; d32 SIMD32 stride → 最多 32 CLs; d32x4 SIMD32 → 8 CLs

3. **L1 命中**: 数据直接从 L1 返回 XVE

4. **L1 未命中**: 请求转发到 L3
   - **L3 命中**: 数据返回 LSC → L1 缓存 (clean) → 返回 XVE
   - **L3 未命中**: 请求发往 DRAM，DRAM 返回数据 → L3 缓存 → LSC → L1 → XVE

5. **XVE 收到数据**, 清除 SBID, 恢复执行

#### Store Path

1. **XVE 发出 store send**（如 `store.ugm.d32.a64`）
   - Store 对 XVE 来说是 fire-and-forget，不等待数据返回

2. **LSC 完全跳过 L1 cache** — store 数据不写入 L1，直接 write-through 到 L3

3. **L3 处理写入**:
   - **Full CL write** (d32 连续): L3 合并数据，标记 CL dirty，无需读 DRAM
   - **Partial CL write** (d16u32 或交错 d32): L3 必须先读取已有 CL（Read-For-Ownership / RFO），再合并 partial 数据。RFO 产生额外的 DRAM 读开销

4. **L3 驱逐**: L3 满时，dirty CL 写回 DRAM; clean CL 直接丢弃

#### 背压模型 (Backpressure)

Read 路径中，每一层都有有限队列。当下游队列满时，上游被阻塞，stall 逐级回传到 XVE：

```
XVE ──send──▶ [rd_tracker queue] ──miss──▶ [L3 superQ] ──miss──▶ [DRAM queue]
     ◀──────  [rd_return_tracker] ◀──hit/fill── L3     ◀──fill──  DRAM
```

当任意队列满时，整条链路向上 stall 直到 XVE。具体 profiler metrics 详见 `hardward-profiler-counter/SKILL.md`。

## 与其他 Skill 的关系

- **hardware-property-query**: 查询实际设备的内存容量、带宽等参数
- **hardware-projection**: 使用内存带宽做 roofline 分析时，需要理解各层级带宽
- **debug-pytorch-xpu-issues**: OOM 等内存相关问题的诊断需要理解内存层级
