# Skill: Dump Intel GPU Kernel ASM

从 SYCL kernel 获取 Intel GPU (Xe/Xe2) 的 Gen ISA 汇编。

## 方法概览

| 方法 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 1. 独立编译 + IGC dump | 能将 kernel 提取为独立 .cpp | 最可靠，kernel 名清晰 | 需要手动剥离依赖 |
| 2. ocloc 编译 SPIR-V + IGC dump | 有 .so 但无法运行 | 不需要运行程序 | kernel 名是编号，需要从 zeinfo 映射 |
| 3. zebin 提取 + IGA 反汇编 | 有 AOT 编译的 .so | 不依赖 IGC dump | 只能拿到 AOT 编译的 kernel（可能只有 double） |
| 4. PyTorch 运行时 IGC dump | 直接从 PyTorch 运行获取 | 最直接 | **实测不工作**（原因未知） |

**推荐优先级**: 方法 1 > 方法 2 > 方法 3。方法 4 目前不可用。

---

## 方法 1: 独立编译 + IGC dump（推荐）

将目标 kernel functor 从源码中提取到独立 `.cpp` 文件，用 `icpx` AOT 编译并通过 IGC 环境变量 dump ASM。

### 步骤

#### 1.1 提取 kernel 源码

从原始 `.cpp` 中复制 kernel functor struct 和 launch 函数。替换 PyTorch 依赖：

```cpp
#include <sycl/sycl.hpp>
#include <cmath>
#include <limits>

// 替换 at::native::memory::aligned_vector
template <typename T, int N>
struct alignas(sizeof(T) * N) aligned_vector {
  T val[N];
  T& operator[](int i) { return val[i]; }
  const T& operator[](int i) const { return val[i]; }
};

// 粘贴 kernel functor struct ...

// 显式实例化：写一个 launch 函数触发模板实例化
void launch(const sycl::half* in, sycl::half* out,
            int dim_size, int outer_size, sycl::queue& q) {
  // 用具体模板参数实例化 kernel functor
  using KernelClass = MyKernelFunctor<8, sycl::half, sycl::half, float, ...>;
  auto kfn = KernelClass(in, out, dim_size, local_size);
  q.submit([&](sycl::handler& h) {
    h.parallel_for<KernelClass>(
        sycl::nd_range<1>(global, local), kfn);
  });
}
```

关键：
- 不需要 `main()` 函数（链接会报错但 IGC 已经 dump 了）
- 需要用 `h.parallel_for<KernelClass>(...)` 显式指定 kernel 类型
- 每种数据类型（fp16, fp32, fp64）需要单独实例化

#### 1.2 编译并 dump

```bash
source /path/to/env.sh  # 初始化 oneAPI

mkdir -p asm_output
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=$(pwd)/asm_output

# AOT 编译到目标设备（bmg = Arc B-series）
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 \
    my_kernel.cpp -o my_kernel.bin 2>&1

# 链接报 "undefined reference to main" 是正常的，IGC 已经完成 dump
```

常用 `-device` 值：

| 设备 | `-device` 值 |
|---|---|
| Arc A-series (DG2/Alchemist) | `dg2` |
| Arc B-series (Battlemage) | `bmg` |
| Data Center Max (PVC) | `pvc` |
| Meteor Lake iGPU | `mtl` |

完整列表: `ocloc compile --help`

#### 1.3 识别 ASM 文件

dump 目录下会生成多个文件：

```
asm_output/
├── OCL_asm<hash>_simd32_entry_0001.asm      # ← kernel 1 的 Gen ASM
├── OCL_asm<hash>_simd32_entry_0001.isaasm    # vISA ASM
├── OCL_asm<hash>_simd32_entry_0001.visa.ll   # vISA LLVM IR
├── OCL_asm<hash>_simd32_entry_0002.asm      # ← kernel 2 的 Gen ASM
├── OCL_asm<hash>.zeinfo                      # kernel 名 → entry 编号映射
├── OCL_asm<hash>_codegen.ll                  # LLVM IR (IGC backend input)
├── OCL_asm<hash>_afterUnification.ll         # LLVM IR (after unification)
├── OCL_asm<hash>_simd16_Intel_Symbol_Table_Void_Program.asm  # 符号表
└── ...
```

**`.asm` 文件就是 Gen ISA 汇编**（硬件指令级别）。

**kernel 名映射**：打开 `.zeinfo` 文件，`kernels` 列表中的顺序对应 `entry_0001`, `entry_0002`, ...

```yaml
# .zeinfo 示例
kernels:
  - name: _ZTS27MyKernelFunctorILi8E...float...   # → entry_0001
  - name: _ZTS27MyKernelFunctorILi4E...half...     # → entry_0002
```

用 `c++filt` 反 mangle kernel 名：
```bash
echo '_ZTS27MyKernelFunctorILi8E...' | c++filt
```

#### 1.4 有用的 IGC 环境变量

| 变量 | 作用 |
|---|---|
| `IGC_ShaderDumpEnable=1` | 启用 dump |
| `IGC_DumpToCustomDir=<path>` | dump 输出目录 |
| `IGC_ShowFullVectorsInShaderDumps=1` | 在 dump 中显示完整 vector（否则超过 1000 元素会截断） |

---

## 方法 2: 从 .so 提取 SPIR-V + ocloc 编译

当无法将 kernel 提取为独立 `.cpp` 时，可以从编译好的 `.so` 中提取 SPIR-V，再用 `ocloc` 编译获取 ASM。

### 步骤

#### 2.1 查看 .so 中的 offload bundle section

```bash
objdump -h my_kernels.so | grep -i "CLANG_OFFLOAD_BUNDLE\|spir"
```

典型输出：
```
 14 __CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen  ...   # AOT 编译的 zebin（特定设备）
 15 __CLANG_OFFLOAD_BUNDLE__sycl-spir64      ...   # SPIR-V（设备无关）
```

- `spir64`: 通用 SPIR-V，可以用 `ocloc` 为任意设备编译
- `spir64_gen`: 已 AOT 编译的 zebin，包含特定设备的原生代码

#### 2.2 提取 SPIR-V section

```bash
objcopy --dump-section \
    "__CLANG_OFFLOAD_BUNDLE__sycl-spir64=raw_spir64.bin" \
    my_kernels.so
```

#### 2.3 检查并解压

提取出的内容可能是：

| Magic bytes | 格式 | 处理方式 |
|---|---|---|
| `03 02 23 07` | 原始 SPIR-V | 直接用 |
| `28 b5 2f fd` | zstd 压缩 | 需要先解压 |
| `7f 45 4c 46` | ELF | 这是 zebin，不是 SPIR-V |

```bash
# 检查 magic
xxd raw_spir64.bin | head -1

# 如果是 zstd 压缩
python3 -c "
import zstandard as zstd
data = open('raw_spir64.bin', 'rb').read()
dec = zstd.ZstdDecompressor().decompress(data)
open('kernels.spv', 'wb').write(dec)
print(f'Decompressed: {len(dec)} bytes, magic: {dec[:4].hex()}')
"
```

> 注意：`zstd` 命令行工具可能无法解压（报 "unsupported format"），用 python `zstandard` 库更可靠。

#### 2.4 用 ocloc 编译 + IGC dump

```bash
mkdir -p ocloc_out
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=$(pwd)/ocloc_out

ocloc compile -file kernels.spv -spirv_input -device bmg -out_dir ocloc_out
```

#### 2.5 识别 kernel

SPIR-V 中所有 kernel 都会被编译。用 `.zeinfo` 文件映射 entry 编号到 kernel 名（同方法 1.3）。

### 注意事项

- **SPIR-V 中可能只包含部分类型的 kernel 实例化**。例如 PyTorch 的 `SoftMaxKernels.so` 的 SPIR-V 中只有 double 类型的 `DispatchSoftmaxForwardKernelFunctor`，float/half 类型的 `SoftmaxForwardKernelFunctor` 不在其中（原因未知，可能是不同编译单元或 JIT 编译）。遇到这种情况需要用方法 1。

- 可以用 `strings kernels.spv | grep "KernelFunctor" | sort -u` 检查 SPIR-V 中包含哪些 kernel。

---

## 方法 3: 从 zebin 提取 + IGA 反汇编

从 `.so` 的 `spir64_gen` section 提取 AOT 编译的 zebin，再用 IGA 反汇编。

### 步骤

#### 3.1 提取 spir64_gen section

```bash
objcopy --dump-section \
    "__CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen=native.bin" \
    my_kernels.so
```

#### 3.2 解压（通常是 zstd → ar archive）

```bash
# 解压 zstd
python3 -c "
import zstandard as zstd
data = open('native.bin', 'rb').read()
dec = zstd.ZstdDecompressor().decompress(data)
open('native.a', 'wb').write(dec)
"

# 提取 ar archive
mkdir ar_contents && cd ar_contents
ar x ../native.a
```

ar 中通常包含：
- `64.bmg` 或 `64.<device>`: 目标设备的 zebin ELF
- `generic_ir`: 通用 IR
- 其他设备变体

#### 3.3 提取 kernel binary 并反汇编

需要用脚本解析 zebin ELF，提取 `.text.` section，然后用 IGA 反汇编：

```bash
# IGA 路径（在 oneAPI 安装目录下）
IGA=/opt/intel/oneapi/compiler/latest/bin/iga64

# 从 zebin 中提取 .text.<kernel_name> section
objcopy --dump-section ".text.MyKernel=kernel.bin" 64.bmg

# IGA 反汇编
$IGA -d -p xe2 kernel.bin > kernel.asm
```

IGA `-p` 平台参数：

| 设备 | `-p` 值 |
|---|---|
| DG2 / Arc A | `xe-hpg` |
| BMG / Arc B | `xe2` |
| PVC | `xe-hpc` |

### 局限

- 只能获取 AOT 编译的 kernel，JIT 编译的 kernel 不在其中
- IGA 反汇编结果没有 IGC dump 的 `.asm` 详细（缺少注释、源码映射等）

---

## 方法 4: PyTorch 运行时 IGC dump（目前不工作）

理论上可以在运行 PyTorch 时通过 IGC 环境变量 dump JIT 编译的 kernel ASM。

```bash
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=$(pwd)/igc_dump
export SYCL_CACHE_PERSISTENT=0    # 禁用持久化 cache，强制重新编译
export SYCL_CACHE_IN_MEM=0        # 禁用内存 cache

python3 -c "
import torch
x = torch.randn(1024, 1024, device='xpu', dtype=torch.float16)
y = torch.nn.functional.softmax(x, dim=-1)
torch.xpu.synchronize()
"
```

**实测结果**: 在当前环境中（IGC `libigdfcl.so.2.32.7`），PyTorch 运行时不会触发 IGC dump（0 个文件生成），即使确认 `libigdfcl.so.2` 被加载（通过 strace 验证）。

相同的 IGC dump 机制在 `ocloc compile` 和独立 SYCL 程序中正常工作。原因可能是 PyTorch/SYCL runtime 的 kernel cache 机制绕过了 IGC 的 dump hook。

---

## ASM 阅读要点

### 关键 store 指令

| 指令 | 含义 | 每 lane 写入 |
|---|---|---|
| `store.ugm.d8u32.a64` | 8-bit in 32-bit lane | 1 byte |
| `store.ugm.d16u32.a64` | 16-bit in 32-bit lane | 2 bytes |
| `store.ugm.d32.a64` | 32-bit | 4 bytes |
| `store.ugm.d32x2.a64` | 32-bit × 2 | 8 bytes |
| `store.ugm.d32x4.a64` | 32-bit × 4 | 16 bytes |
| `store.ugm.d64.a64` | 64-bit | 8 bytes |

SIMD32 下每条 store message 总写入 = 每 lane 写入 × 32 lanes。

`ugm` = Untyped Global Memory (通过 LSC)
`slm` = Shared Local Memory
`a64` = 64-bit 地址

### 关键 load 指令

格式同 store，但用 `load` 替代 `store`。

### 寄存器表示

```
r4:2     → r4 和 r5（2 个连续 GRF）
r8:8     → r8 到 r15（8 个连续 GRF）
r4.0<1>:f   → r4，从 offset 0 开始，stride 1，float 类型
r9.0<2>:hf  → r9，stride 2（隔一个写一个），half 类型
r32.1<2>:uw → r32，从 offset 1 开始，stride 2，unsigned word (16-bit)
```

### IGC store merging 行为

IGC 会尝试将多条连续地址的 store 合并为更宽的 store（如 4× d32 → 1× d32x4）。
合并成功的条件：
1. 地址连续
2. **源寄存器连续**（如 r8, r9, r10, r11 → r8:4）
3. 中间没有打断的指令

**已知问题**: 当数据类型需要 pack/unpack（如 float→half 后打包到 dword）时，
pack 指令会导致源寄存器不连续，阻止 store merging。
此时需要在源码中显式使用 vectorized store（如 `aligned_vector` reinterpret_cast）来绕过。

---

## 实战案例

### Softmax FP16 slow path vec store 优化

**场景**: `SoftmaxForwardKernelFunctor`，vec_size=8，half→half with float accumulator

**问题**: main branch 逐元素写 `out_data_p[j] = result`，IGC 将 8 个 half 打包成 4 个 dword，
生成 4× `store.ugm.d32.a64`。但 half→dword pack（stride-2 mov + uw interleave）
导致源寄存器不连续（r4, r32, r12, r6），IGC 无法合并为 d32x4。

**对比**: FP32 相同结构的代码，IGC 成功合并为 1× `store.ugm.d32x4.a64`（float 计算结果
自然排列在连续寄存器中，无需 pack）。

**修复**: 源码中用 `*(reinterpret_cast<out_vec_t*>(...)) = ...` 显式 vectorized store，
编译器直接生成 1× `store.ugm.d32x4.a64`。

**效果**: FP16 softmax ~1.9x 加速（L3_WRITE -75%, L3_STALL 70%→0%, RFO 消除）。

**分析文件**:
- ASM: `temp/softmax_asm/{main,vecstore}_{fp16,fp32}.asm`
- Counter 对比: `temp/softmax_vec_store_counters.md`
- 实验记录: `xpu-perf/micro_perf/softmax.md`
