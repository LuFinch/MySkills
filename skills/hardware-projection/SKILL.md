# Hardware Performance Projection Skill

Perform theoretical performance ceiling analysis (Roofline model) for operators or models based on hardware specifications.

> Compatible with: Claude Code, OpenCode, GitHub Copilot, Roo Code, and other AI coding assistants.

## Overview

This skill guides AI agents in performing theoretical hardware performance analysis, including:

0. Getting Compute Throughput and Memory Bandwidth for target device
1. Building a Roofline model from hardware parameters
2. Analyzing operator Arithmetic Intensity
3. Determining whether an operator is compute-bound or memory-bound
4. Calculating theoretical performance ceilings and actual efficiency

**Prerequisites**: Use the `hardware-property-query` skill to obtain device properties. This skill focuses on analysis and projection, not raw property querying.

## Directory Structure

```
hardware-projection/
├── SKILL.md                # This file
└── xpu_models/             # Device performance profiles (specs + measured data)
```

## Compute Throughput Formulas

> **All formulas below require architecture-specific parameters that cannot be queried from SYCL.** The agent must ask the user or look up official specs before calculating.

### FP32 Vector Engine

```
Peak FP32 VE GFLOPS = EU_count × FP32_ALUs_per_EU × 2 (FMA) × clock_MHz / 1000
```

- `FP32_ALUs_per_EU`: architecture-dependent, **ask the user**
- `EU_count` and `clock_MHz`: can be queried via `hardware-property-query` skill (but `SIMD_width` ≠ `FP32_ALUs_per_EU`)

### FP16 Vector Engine

```
Case A — ALU has dedicated packed-FP16 support:
  Peak FP16 VE GFLOPS = Peak FP32 VE GFLOPS × 2
  (Each 32-bit ALU datapath processes 2 × FP16 ops in parallel per cycle.
   Requires dedicated circuit design — NOT automatic.)

Case B — ALU does NOT have packed-FP16 support:
  Peak FP16 VE GFLOPS = Peak FP32 VE GFLOPS
  (FP16 promoted to FP32; no compute speedup, only memory bandwidth savings)
```

- **Which case applies?** Architecture-dependent — **ask the user**.

### FP16/BF16 XMX Engine (Matrix Operations)

On architectures with XMX support (e.g., Intel Xe-HPG, Xe-HPC, Xe2), each EU typically contains **one Vector Engine and one XMX Engine** as separate execution units.

```
Peak FP16 XMX GFLOPS = EU_count × OPs_per_XMX_per_cycle_per_EU × 2 (FMA) × clock_MHz / 1000
```

- `OPs_per_XMX_per_cycle_per_EU`: architecture-dependent, **not queryable from SYCL**
- **In practice**: look up the official spec directly — do not attempt to derive from VE FP32
- Not all architectures have XMX (e.g., Xe-LPG iGPU may lack it) — **ask the user** to confirm

### Summary Table

| Metric | Formula | User Must Provide |
|--------|---------|-------------------|
| FP32 VE | `EU × FP32_ALUs_per_EU × 2(FMA) × clock` | `FP32_ALUs_per_EU` |
| FP16 VE (packed) | `FP32_VE × 2` | Confirm ALU has dedicated packed-FP16 circuits |
| FP16 VE (no packed) | `= FP32_VE` (promote to FP32) | Confirm ALU lacks packed-FP16 support |
| FP16/BF16 XMX | Official spec value or Ask User | XMX peak TFLOPS from datasheet |

**Why ×2 (FMA)?** FMA (Fused Multiply-Add) computes `a × b + c` in one cycle, producing **2 FLOPs** per ALU. Nearly all vendor-published TFLOPS specs assume FMA. If a kernel only uses pure add or pure multiply, effective peak is halved.

### Xe2 / BMG Architecture Example (e.g., Arc B580)

Each EU contains **one Vector Engine + one XMX Engine**:

**Vector Engine** has two execution ports:

| Port | Functional Units | Description |
|------|-----------------|-------------|
| Port 0 | 16 × FP32 | 16 FP32 ALUs for floating-point operations |
| Port 1 | 16 × INT32 + 4 × SFU | 16 integer ALUs + 4 Special Function Units |

- Port 0 provides **16 FP32 ops/cycle/EU**, so `FP32_ALUs_per_EU = 16`

Verification (Arc B580: 160 EUs, 2850 MHz):
```
160 × 16 × 2(FMA) × 2850 / 1000 = 14592 GFLOPS ≈ 14.6 TFLOPS ✓
```

## Memory Bandwidth

### From SYCL Query

If `ext_intel_max_mem_bandwidth` is available, use it directly (reported in bytes/s).

### From Official Specs

```
Peak BW (GB/s) = speed_Gbps × total_bus_width_bits / 8
```

**Critical**: For HBM devices, `total_bus_width` must be the **aggregate across all stacks/channels**, not a single channel's width.
- A100: 5 HBM2e stacks × 1024-bit = 5120-bit total
- PVC Max 1550: 8 HBM2e stacks × 1024-bit = 8192-bit total

> **Known issue**: SYCL's `ext_intel_memory_bus_width` reports **per-channel** width (e.g., 64 bits), not total. Do NOT use it to calculate aggregate bandwidth.

> These are approximate values from public specs. The agent should prefer measured profiles from `xpu_models/`.

## Core Concepts

### Roofline Model

```
Performance (FLOPS) = min(Peak_Compute, Peak_Bandwidth × Arithmetic_Intensity)
```

- **Peak Compute (FLOPS)**: Hardware peak compute power
- **Peak Bandwidth (Bytes/s)**: Memory/VRAM bandwidth
- **Arithmetic Intensity (FLOP/Byte)**: Computation per byte of data transferred
- **Ridge Point**: `Peak_Compute / Peak_Bandwidth`, the crossover point

### Determining Bound Type

```
AI = Total_FLOPs / Total_Bytes_Transferred

# Step 1: Determine which hardware path the kernel uses
#   Element-wise, reduction, etc. → Vector Engine → use Peak_VE
#   GEMM, convolution, attention  → XMX Engine   → use Peak_XMX
Ridge_Point = Peak_Compute_for_this_path / Peak_Bandwidth

# Step 2: Compare
if AI < Ridge_Point:
    → Memory-Bound
else:
    → Compute-Bound
```

## Hardware Parameter Template

| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| Peak VE Compute (FP32) | `P_VE_fp32` | TFLOPS | Single-precision peak |
| Peak VE Compute (FP16/BF16) | `P_VE_fp16` | TFLOPS | Half-precision peak |
| Peak XMX Compute (FP16/BF16) | `P_XMX_fp16` | TFLOPS | Half-precision peak |
| Memory Bandwidth | `BW_mem` | GB/s | HBM/GDDR bandwidth |
| L2 Cache Bandwidth | `BW_l2` | GB/s | If known |
| Memory Capacity | `Mem` | GB | — |
| Compute Unit Count | `CU/SM/EU` | — | Platform-specific |
| Clock Frequency | `Freq` | GHz | Boost frequency |

## Analysis Workflow

### Step 1: Collect Hardware Parameters

Priority order:
1. Check if there is a profile for the target device under `xpu_models/`
    1.1 if yes, use the profile
    1.2 if no, run the `hardware-property-query` skill's device query tool for live data
3. **For all compute throughput**: Ask the user for ALU-per-EU count, native precision support, and XMX specs — or look up official device specs
4. Ask the user or obtain from public specs for any missing values
5. Save the `Hardware Parameter Template` and `Ridge_Point` into profile
6. **Update device profile**: If during the analysis you calculated or obtained concrete values for compute throughput (e.g., FP32 VE, FP16/BF16 XMX TFLOPS) or memory bandwidth (GB/s) — whether from official specs, user input, or derivation — update the corresponding device profile under `xpu_models/` to fill in previously missing values. This ensures future analyses can reuse these values without re-querying.

### Step 2: Analyze the Operator/Kernel

Calculate for the target operator:

1. **Total Computation (FLOPs)**
2. **Total Data Movement (Bytes)** — including input reads + output writes
3. **Arithmetic Intensity AI = FLOPs / Bytes**

### Step 3: Common Operator Computation Formulas

#### GEMM / MatMul: `[M, K] × [K, N] → [M, N]`

```
FLOPs = 2 × M × K × N
Bytes = (M×K + K×N + M×N) × sizeof(dtype)
AI = 2×M×K×N / ((M×K + K×N + M×N) × sizeof(dtype))
```

When M, K, N are all large: `AI ≈ min(M, N, K) / sizeof(dtype)`

#### Element-wise Op (e.g., ReLU, Add): `[N]`

```
FLOPs = N  (or 2N for fused ops)
Bytes = 2 × N × sizeof(dtype)   (read + write)
AI = 0.5 / sizeof(dtype)        (very low, always memory-bound)
```

#### Reduction (e.g., Softmax, LayerNorm): `[M, N]`

```
FLOPs ≈ 5 × M × N   (softmax: exp + sum + div, approximate)
Bytes ≈ 2 × M × N × sizeof(dtype)
AI ≈ 2.5 / sizeof(dtype)       (typically memory-bound)
```

#### Attention (SDPA): `Q[B,H,S,D], K[B,H,S,D], V[B,H,S,D]`

**Standard implementation:**
```
FLOPs = 4 × B × H × S² × D
Bytes = B×H × (3×S×D + S×S + S×D) × sizeof(dtype)
```

**Flash Attention (fused kernel):**
```
FLOPs = 4 × B × H × S² × D        (same)
Bytes ≈ B×H × (3×S×D + S×D) × sizeof(dtype)  (no S×S intermediate)
AI significantly improves, especially when S >> D
```

#### Convolution 2D: Input `[N,C_in,H,W]`, Kernel `[C_out,C_in,kH,kW]`

```
FLOPs = 2 × N × C_out × H_out × W_out × C_in × kH × kW
Bytes = (N×C_in×H×W + C_out×C_in×kH×kW + N×C_out×H_out×W_out) × sizeof(dtype)
```

### Step 4: Calculate Theoretical Performance Ceiling

```python
time_compute = total_flops / peak_compute   # seconds
time_memory  = total_bytes / peak_bandwidth  # seconds
time_theory  = max(time_compute, time_memory)

if time_memory > time_compute:
    bound_type = "Memory-Bound"
    achieved_bw = total_bytes / actual_time
    efficiency  = achieved_bw / peak_bandwidth × 100%
else:
    bound_type = "Compute-Bound"
    achieved_flops = total_flops / actual_time
    efficiency     = achieved_flops / peak_compute × 100%
```

> **Do NOT** use `total_flops / time_theory` as universal "theoretical throughput" — for memory-bound kernels the bottleneck is bandwidth, not compute.

### Step 5: Output Report Format

```
## Performance Projection Report

### Hardware
- Device: <name>
- Hardware Path: <Vector Engine / XMX Engine>
- Peak Compute: <X> TFLOPS (specify VE or XMX, and precision)
- Memory Bandwidth: <Y> GB/s
- Ridge Point: <Z> FLOP/Byte (for the hardware path used)
- Source: <xpu_models/xxx.md | live query | user provided>

### Operator: <op_name>
- Shape: <input shapes>
- Total FLOPs: <N>
- Total Bytes: <M>
- Arithmetic Intensity: <AI> FLOP/Byte
- Bound Type: <Compute-Bound / Memory-Bound>

### Projection
- Theoretical Time: <T> ms
- If Compute-Bound: Theoretical Throughput = <X> TFLOPS, Efficiency = achieved_FLOPS / peak_compute
- If Memory-Bound: Theoretical Bandwidth = <Y> GB/s, Efficiency = achieved_BW / peak_BW

### Optimization Suggestions
- <suggestions based on bound type>
```

## Optimization Suggestion Templates

### Memory-Bound Operators

1. **Kernel Fusion**: Reduce intermediate results written back to VRAM
2. **Use Lower Precision**: FP16/BF16/INT8 to reduce data movement
3. **Improve Data Reuse**: Leverage shared memory / L2 cache tiling
4. **Flash Attention and Other Fused Kernels**: Avoid materializing intermediate matrices

### Compute-Bound Operators

1. **Use Tensor Cores / XMX**: Ensure hardware matrix acceleration units are utilized
2. **Optimize Tile Size**: Match hardware compute unit count
3. **Improve Occupancy**: Adjust register/shared memory usage
4. **Lower Precision Compute**: FP16 compute is typically 2x+ of FP32

## Multi-Level Roofline

For hardware with cache hierarchies:

```
Level 1: HBM/GDDR Roofline → BW = VRAM bandwidth
Level 2: L2 Roofline       → BW = L2 bandwidth (typically 2-4x VRAM)
Level 3: L1/SLM Roofline   → BW = L1/shared memory bandwidth
```

A well-tiled kernel can complete most memory accesses at L2/L1 level, exceeding the VRAM roofline limit.

## Model-Level Analysis

When projecting for an entire model:

1. Analyze each operator layer by layer for bound type and theoretical time
2. Sum up to get total theoretical time
3. Consider kernel launch overhead (typically ~5-10us per launch, but may be overlaped due to async kernel launch)
4. Consider host-device synchronization overhead
5. Account for pipeline bubbles

```python
total_time = sum(max(flops_i/peak_compute, bytes_i/peak_bw) for each layer_i)
total_time += num_kernels × kernel_launch_overhead
```

## Comparison with Actual Measurements

| Efficiency Range | Rating | Possible Causes |
|-----------------|--------|-----------------|
| >80% | Excellent | Close to hardware limit |
| 60-80% | Good | Minor optimization opportunities |
| 30-60% | Fair | Obvious bottlenecks (launch overhead, low occupancy, bank conflicts) |
| <30% | Poor | Serious issues (data format mismatch, acceleration units unused, excessive sync) |

### Bandwidth-Bound Quick Reference

| Operator | Typical AI (FP32) | Bound |
|----------|-------------------|-------|
| SoftMax (3 pass read + 1 write) | ~0.25 FLOP/Byte | BW-bound |
| LayerNorm (2 pass read + 1 write) | ~0.5 FLOP/Byte | BW-bound |
| ReduceSum (1 read) | ~0.25 FLOP/Byte | BW-bound |
| TopK (radix, multi-pass) | varies | mostly BW-bound |
| GEMM (large) | >> 100 FLOP/Byte | Compute-bound |

Peak element throughput for BW-bound ops:
- **FP32**: Peak_BW / 4 bytes/element
- **FP16**: Peak_BW / 2 bytes/element

## How to Add a New Device Profile

1. Run the `hardware-property-query` tool to get device properties
2. Look up official specs for compute throughput (FP32/FP16/INT8, VE and XMX) and memory bandwidth
3. Create a file under `xpu_models/`, naming format: `<device_name>_<vendor_id>_<device_id>.md`
4. Fill in: Device Identity → Compute Properties → Memory Properties → Peak Performance Projections → Roofline Crossover

### Profile Template

```markdown
# <Device Full Name>

## Device Identity
- **Device Name**: <name>
- **PCI Device ID**: 0x<id>
- **Architecture**: <arch>
- **Driver Version**: <version>

## Compute Properties
| Property | Value |
|----------|-------|
| Execution Units (EUs) | <count> |
| EU SIMD Width | <width> |
| GPU Slices | <count> |
| Subslices per Slice | <count> |
| EUs per Subslice | <count> |
| HW Threads per EU | <count> |
| Max Clock Frequency | <MHz> |
| Sub-group Sizes | <sizes> |
| Max Work-group Size | <size> |

## Memory Properties
| Property | Value |
|----------|-------|
| Global Memory (VRAM) | <size> |
| Local Memory (SLM) | <size> per work-group |
| Global Mem Cache (L2) | <size> |
| Cache Line Size | <bytes> |
| Memory Bus Width | <spec value>-bit |
| Memory Speed | <Gbps> |

## Peak Performance Projections

### Compute Throughput
| Precision | Path | Peak Throughput |
|-----------|------|----------------|
| FP32 | Vector Engine | **X TFLOPS** |
| FP16 | Vector Engine | **? TFLOPS** |
| FP16/BF16 | XMX Engine | **Y TFLOPS** |
| INT8 | XMX Engine | **Z TOPS** |

### Memory Bandwidth
| Metric | Value |
|--------|-------|
| Peak BW | **W GB/s** |

### Arithmetic Intensity Crossover
| Kernel Type | Hardware Path | Crossover (FLOP/Byte) |
|-------------|---------------|----------------------|
| Element-wise, reduction | VE FP32 | Peak_FP32_VE / Peak_BW |
| Element-wise, reduction | VE FP16 | Peak_FP16_VE / Peak_BW |
| GEMM, convolution, attention | XMX FP16/BF16 | Peak_FP16_XMX / Peak_BW |
| GEMM, convolution, attention | XMX INT8 | Peak_INT8_XMX / Peak_BW |
```

## Agent Workflow for Compute Throughput

1. Check if the device profile under `xpu_models/` has the value
2. If not, **ask the user**:
   - "How many FP32 ALUs per EU does this architecture have?" (for VE FP32)
   - "Does the ALU natively support FP16?" (for VE FP16)
   - "What is the XMX peak throughput for FP16/BF16/INT8?" (for matrix ops)
3. If the user doesn't know, suggest looking up the official device spec or architecture whitepaper
4. **Never silently assume** ALU counts from SYCL `SIMD_width`, or derive FP16/INT8 from FP32 by multiplying
