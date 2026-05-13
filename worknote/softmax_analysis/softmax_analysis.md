# Softmax Slow Path Vec Store 优化 — LSC / L3 / DRAM Counter 对比

> 设备: Intel Arc Pro B60 | Unitrace `-q` 模式 | Benchmark: `softmax_bench.py --profile`
>
> - **main**: PyTorch main branch (`pytorch/main_dist/torch-2.13.0a0+git0290cd4`)
> - **vec**: 添加 softmax vec store 优化 (`pytorch/dist/torch-2.13.0a0+git0290cd4`)

---

## Reference: GpuTime

| Counter | fp16 [1024,32768] | fp16 [1024,65536] | fp16 [1024,131072] | fp32 [1024,32768] | fp32 [1024,65536] | fp32 [1024,131072] |
|---|---|---|---|---|---|---|
| `GpuTime[ns]` [main] | 778,544 | 1,610,284 | 3,395,236 | 728,884 | 1,597,752 | 4,918,368 |
| `GpuTime[ns]` [vec] | 546,104 | 1,121,068 | 2,292,992 | 813,696 | 1,788,124 | 4,948,632 |
| diff | **-29.9%** | **-30.4%** | **-32.5%** | +11.6% | +11.9% | +0.6% |

---

## LSC / L1 Cache

| Counter | fp16 [1024,32768] | fp16 [1024,65536] | fp16 [1024,131072] | fp32 [1024,32768] | fp32 [1024,65536] | fp32 [1024,131072] |
|---|---|---|---|---|---|---|
| `LOAD_STORE_CACHE_ACCESS` [main] | 7,372,800 | 14,712,832 | 29,392,896 | 8,421,376 | 16,809,984 | 33,587,200 |
| `LOAD_STORE_CACHE_ACCESS` [vec] | 4,227,072 | 8,421,376 | 16,809,984 | 8,421,376 | 16,809,984 | 33,587,200 |
| diff | **-42.7%** | **-42.8%** | **-42.8%** | +0.0% | +0.0% | +0.0% |
| | | | | | | |
| `LOAD_STORE_CACHE_HIT` [main] | 2,916,049 | 5,411,967 | 9,473,325 | 5,321,386 | 9,472,467 | 18,906,273 |
| `LOAD_STORE_CACHE_HIT` [vec] | 2,916,096 | 5,168,820 | 9,469,642 | 5,183,121 | 9,470,916 | 18,906,235 |
| diff | +0.0% | -4.5% | -0.0% | -2.6% | -0.0% | -0.0% |
| | | | | | | |
| `LOAD_STORE_CACHE_BYTE_READ` [main] | 202,375,168 | 403,701,760 | 806,354,944 | 403,701,760 | 806,354,944 | 1,611,661,312 |
| `LOAD_STORE_CACHE_BYTE_READ` [vec] | 202,375,168 | 403,701,760 | 806,354,944 | 403,701,760 | 806,354,944 | 1,611,661,312 |
| diff | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| | | | | | | |
| `LOAD_STORE_CACHE_BYTE_WRITE` [main] | 0 | 0 | 0 | 0 | 0 | 0 |
| `LOAD_STORE_CACHE_BYTE_WRITE` [vec] | 0 | 0 | 0 | 0 | 0 | 0 |
| diff | N/A | N/A | N/A | N/A | N/A | N/A |
| | | | | | | |
| `LOAD_STORE_CACHE_PARTIAL_WRITE_COUNT` [main] | 0 | 0 | 0 | 0 | 0 | 0 |
| `LOAD_STORE_CACHE_PARTIAL_WRITE_COUNT` [vec] | 0 | 0 | 0 | 0 | 0 | 0 |
| diff | N/A | N/A | N/A | N/A | N/A | N/A |
| | | | | | | |
| `XVE_LSC_READ_MESSAGE_COUNT` [main] | 425,984 | 819,200 | 1,605,632 | 819,200 | 1,605,632 | 3,178,496 |
| `XVE_LSC_READ_MESSAGE_COUNT` [vec] | 425,984 | 819,200 | 1,605,632 | 819,200 | 1,605,632 | 3,178,496 |
| diff | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| | | | | | | |
| `XVE_LSC_WRITE_MESSAGE_COUNT` [main] | 524,288 | 1,048,576 | 2,097,152 | 262,144 | 524,288 | 1,048,576 |
| `XVE_LSC_WRITE_MESSAGE_COUNT` [vec] | 131,072 | 262,144 | 524,288 | 262,144 | 524,288 | 1,048,576 |
| diff | **-75.0%** | **-75.0%** | **-75.0%** | +0.0% | +0.0% | +0.0% |
| | | | | | | |
| `XVE_LSC_REGISTER_REQUEST_COUNT` [main] | 5,308,416 | 10,027,008 | 19,464,192 | 6,881,280 | 13,172,736 | 25,755,648 |
| `XVE_LSC_REGISTER_REQUEST_COUNT` [vec] | 3,735,552 | 6,881,280 | 13,172,736 | 6,881,280 | 13,172,736 | 25,755,648 |
| diff | **-29.6%** | **-31.4%** | **-32.3%** | +0.0% | +0.0% | +0.0% |
| | | | | | | |
| `XVE_LSC_REGISTER_RESPONSE_COUNT` [main] | 3,309,568 | 6,455,296 | 12,746,752 | 6,455,296 | 12,746,752 | 25,329,664 |
| `XVE_LSC_REGISTER_RESPONSE_COUNT` [vec] | 3,309,568 | 6,455,296 | 12,746,752 | 6,455,296 | 12,746,752 | 25,329,664 |
| diff | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| | | | | | | |
| `LSC_INPUT_AVAILABLE[%]` [main] | 19.11 | 12.38 | 8.69 | 8.45 | 6.39 | 5.84 |
| `LSC_INPUT_AVAILABLE[%]` [vec] | 10.90 | 8.73 | 7.55 | 10.80 | 8.81 | 4.41 |
| diff | -43.0% | -29.5% | -13.1% | +27.9% | +37.8% | -24.5% |
| | | | | | | |
| `LSC_OUTPUT_READY[%]` [main] | 3.09 | 2.66 | 2.38 | 3.64 | 3.01 | 2.88 |
| `LSC_OUTPUT_READY[%]` [vec] | 5.11 | 4.22 | 3.73 | 5.18 | 4.32 | 2.12 |
| diff | +65.4% | +58.3% | +56.9% | +42.3% | +43.7% | -26.4% |
| | | | | | | |
| `LSC_BANK_ACCESS_COUNT` [main] | 7,695,651 | 15,035,661 | 29,715,843 | 8,743,646 | 17,132,169 | 33,910,417 |
| `LSC_BANK_ACCESS_COUNT` [vec] | 4,550,093 | 8,744,799 | 17,132,855 | 8,742,306 | 17,131,132 | 33,911,402 |
| diff | **-40.9%** | **-41.8%** | **-42.3%** | -0.0% | -0.0% | +0.0% |

---

## L3 / Device Cache

| Counter | fp16 [1024,32768] | fp16 [1024,65536] | fp16 [1024,131072] | fp32 [1024,32768] | fp32 [1024,65536] | fp32 [1024,131072] |
|---|---|---|---|---|---|---|
| `L3_READ` [main] | 1,055,468 | 3,654,356 | 12,575,932 | 4,013,678 | 12,576,453 | 25,175,577 |
| `L3_READ` [vec] | 1,054,977 | 4,626,634 | 12,590,394 | 4,567,309 | 12,588,822 | 25,176,971 |
| diff | -0.0% | +26.6% | +0.1% | +13.8% | +0.1% | +0.0% |
| | | | | | | |
| `L3_WRITE` [main] | 4,194,313 | 8,388,617 | 16,777,225 | 2,097,161 | 4,194,319 | 8,388,618 |
| `L3_WRITE` [vec] | 1,048,585 | 2,097,163 | 4,194,315 | 2,097,163 | 4,194,319 | 8,388,618 |
| diff | **-75.0%** | **-75.0%** | **-75.0%** | +0.0% | +0.0% | +0.0% |
| | | | | | | |
| `L3_HIT` [main] | 3,150,754 | 7,832,640 | 19,861,962 | 3,477,948 | 10,453,201 | 9,654,944 |
| `L3_HIT` [vec] | 791,202 | 4,074,356 | 10,294,586 | 4,026,327 | 10,569,692 | 9,341,544 |
| diff | **-74.9%** | **-48.0%** | **-48.2%** | +15.8% | +1.1% | -3.2% |
| | | | | | | |
| `L3_MISS` [main] | 2,099,027 | 4,210,333 | 9,491,195 | 2,632,891 | 6,317,571 | 23,909,251 |
| `L3_MISS` [vec] | 1,312,361 | 2,649,441 | 6,490,123 | 2,638,145 | 6,213,449 | 24,224,045 |
| diff | **-37.5%** | **-37.1%** | **-31.6%** | +0.2% | -1.6% | +1.3% |
| | | | | | | |
| `L3_STALL[%]` [main] | 70.50 | 69.60 | 68.85 | 7.68 | 1.81 | 1.18 |
| `L3_STALL[%]` [vec] | 0.03 | 0.01 | 0.02 | 1.55 | 0.81 | 1.12 |
| diff | **-100.0%** | **-100.0%** | **-100.0%** | -79.9% | -55.5% | -5.4% |
| | | | | | | |
| `L3_BUSY[%]` [main] | 98.76 | 99.22 | 99.54 | 98.53 | 99.29 | 99.76 |
| `L3_BUSY[%]` [vec] | 91.54 | 93.03 | 95.28 | 97.45 | 98.62 | 99.79 |
| diff | -7.3% | -6.2% | -4.3% | -1.1% | -0.7% | +0.0% |
| | | | | | | |
| `L3_SUPERQ_FULL[%]` [main] | 12.50 | 12.61 | 12.03 | 1.05 | 0.13 | 0.16 |
| `L3_SUPERQ_FULL[%]` [vec] | 0.00 | 0.00 | 0.00 | 0.14 | 0.09 | 0.17 |
| diff | **-100.0%** | **-100.0%** | **-100.0%** | -87.0% | -32.4% | +7.3% |
| | | | | | | |
| `L3_INPUT_AVAILABLE[%]` [main] | 73.41 | 72.80 | 72.25 | 14.53 | 9.65 | 9.10 |
| `L3_INPUT_AVAILABLE[%]` [vec] | 7.44 | 8.96 | 9.90 | 12.44 | 12.22 | 6.93 |
| diff | **-89.9%** | **-87.7%** | **-86.3%** | -14.3% | +26.7% | -23.9% |
| | | | | | | |
| `L3_OUTPUT_READY[%]` [main] | 9.94 | 10.02 | 10.17 | 7.38 | 7.97 | 8.02 |
| `L3_OUTPUT_READY[%]` [vec] | 7.43 | 8.96 | 9.88 | 11.05 | 11.51 | 5.88 |
| diff | -25.2% | -10.6% | -2.9% | +49.7% | +44.5% | -26.6% |
| | | | | | | |
| `LSC_L3_READ` [main] | 1,048,979 | 3,667,963 | 12,577,987 | 4,630,729 | 12,577,343 | 25,166,598 |
| `LSC_L3_READ` [vec] | 1,048,979 | 4,621,081 | 12,572,209 | 4,003,051 | 12,568,016 | 25,166,737 |
| diff | +0.0% | +26.0% | -0.0% | -13.6% | -0.1% | +0.0% |
| | | | | | | |
| `LSC_L3_WRITE` [main] | 4,194,304 | 8,388,608 | 16,777,216 | 2,097,152 | 4,194,304 | 8,388,608 |
| `LSC_L3_WRITE` [vec] | 1,048,576 | 2,097,152 | 4,194,304 | 2,097,152 | 4,194,304 | 8,388,608 |
| diff | **-75.0%** | **-75.0%** | **-75.0%** | +0.0% | +0.0% | +0.0% |
| | | | | | | |
| `LSC_L3_HIT` [main] | 3,146,129 | 7,849,461 | 19,844,236 | 4,090,540 | 10,526,867 | 9,846,428 |
| `LSC_L3_HIT` [vec] | 786,830 | 4,076,970 | 10,431,455 | 3,466,284 | 10,383,294 | 9,500,343 |
| diff | **-75.0%** | **-48.1%** | **-47.4%** | -15.3% | -1.4% | -3.5% |

---

## DRAM / Global Memory

| Counter | fp16 [1024,32768] | fp16 [1024,65536] | fp16 [1024,131072] | fp32 [1024,32768] | fp32 [1024,65536] | fp32 [1024,131072] |
|---|---|---|---|---|---|---|
| `GPU_MEMORY_BYTE_READ` [main] | 134,427,136 | 269,700,608 | 607,956,480 | 135,180,288 | 337,720,832 | 1,397,096,448 |
| `GPU_MEMORY_BYTE_READ` [vec] | 67,317,760 | 136,267,776 | 348,798,976 | 135,527,424 | 331,045,888 | 1,417,247,744 |
| diff | **-49.9%** | **-49.5%** | **-42.6%** | +0.3% | -2.0% | +1.4% |
| | | | | | | |
| `GPU_MEMORY_BYTE_WRITE` [main] | 75,753,472 | 143,363,584 | 277,545,984 | 143,432,704 | 277,422,080 | 547,152,384 |
| `GPU_MEMORY_BYTE_WRITE` [vec] | 75,902,464 | 143,512,064 | 277,611,008 | 143,410,176 | 277,902,848 | 547,394,560 |
| diff | +0.2% | +0.1% | +0.0% | -0.0% | +0.2% | +0.0% |
| | | | | | | |
| `GPU_MEMORY_BYTE_READ_RATE[GBpS]` [main] | 172.66 | 167.49 | 179.06 | 185.46 | 211.37 | 284.06 |
| `GPU_MEMORY_BYTE_READ_RATE[GBpS]` [vec] | 123.27 | 121.55 | 152.12 | 166.56 | 185.14 | 286.39 |
| diff | -28.6% | -27.4% | -15.0% | -10.2% | -12.4% | +0.8% |
| | | | | | | |
| `GPU_MEMORY_BYTE_WRITE_RATE[GBpS]` [main] | 97.30 | 89.03 | 81.75 | 196.78 | 173.63 | 111.25 |
| `GPU_MEMORY_BYTE_WRITE_RATE[GBpS]` [vec] | 138.99 | 128.01 | 121.07 | 176.25 | 155.42 | 110.62 |
| diff | +42.8% | +43.8% | +48.1% | -10.4% | -10.5% | -0.6% |
| | | | | | | |
| `GPU_MEMORY_64B_TRANSACTION_READ` [main] | 1,050,214 | 2,107,035 | 4,749,661 | 1,056,100 | 2,638,448 | 10,914,810 |
| `GPU_MEMORY_64B_TRANSACTION_READ` [vec] | 525,916 | 1,064,597 | 2,724,987 | 1,058,801 | 2,586,298 | 11,072,250 |
| diff | **-49.9%** | **-49.5%** | **-42.6%** | +0.3% | -2.0% | +1.4% |
| | | | | | | |
| `GPU_MEMORY_64B_TRANSACTION_WRITE` [main] | 591,826 | 1,120,025 | 2,168,321 | 1,120,578 | 2,167,357 | 4,274,626 |
| `GPU_MEMORY_64B_TRANSACTION_WRITE` [vec] | 592,988 | 1,121,194 | 2,168,835 | 1,120,395 | 2,171,120 | 4,276,525 |
| diff | +0.2% | +0.1% | +0.0% | -0.0% | +0.2% | +0.0% |
| | | | | | | |
| `GPU_MEMORY_L3_READ` [main] | 1,310,823 | 2,625,130 | 5,518,304 | 527,013 | 1,317,149 | 5,452,998 |
| `GPU_MEMORY_L3_READ` [vec] | 262,242 | 531,038 | 1,360,212 | 528,367 | 1,295,407 | 5,531,653 |
| diff | **-80.0%** | **-79.8%** | **-75.4%** | +0.3% | -1.7% | +1.4% |
| | | | | | | |
| `GPU_MEMORY_L3_WRITE` [main] | 295,211 | 558,768 | 1,081,955 | 556,078 | 1,078,410 | 2,134,241 |
| `GPU_MEMORY_L3_WRITE` [vec] | 295,687 | 559,327 | 1,081,873 | 557,308 | 1,081,344 | 2,134,638 |
| diff | +0.2% | +0.1% | -0.0% | +0.2% | +0.3% | +0.0% |
| | | | | | | |
| `GPU_MEMORY_REQUEST_QUEUE_FULL[%]` [main] | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `GPU_MEMORY_REQUEST_QUEUE_FULL[%]` [vec] | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| diff | N/A | N/A | N/A | N/A | N/A | N/A |
| | | | | | | |
| `TLB_MISS` [main] | 1,937 | 2,548 | 2,883 | 3,142 | 3,403 | 3,624 |
| `TLB_MISS` [vec] | 2,212 | 3,051 | 3,504 | 3,226 | 3,546 | 3,479 |
| diff | +14.2% | +19.7% | +21.5% | +2.7% | +4.2% | -4.0% |

---

## 关键发现

### FP16: vec store 消除 4×d32 partial write

#### 根因（ASM 确认）

main branch 的 `SoftmaxForwardKernelFunctor` 热路径中，编译器将 `out_data_p[j] = result`
（8 个 half 逐元素写）优化为 **4 条 `store.ugm.d32.a64`**（2 个 half 打包为 1 个 dword），
而非 8 条 `d16u32`。热路径没有 `d16u32`。

问题在于：4 条独立的 d32 store 写到同一个 64B cache line 的不同 1/4 区域，
每条只覆盖 4B × 32 lanes = 128B，L3 看到 4 次 partial write。

Vec store branch 用 1 条 `store.ugm.d32x4.a64` 替代，16B × 32 lanes = 512B，
一次写满整个 CL。

ASM 证据（详见下方 §ASM 对比）：

```
main:     4× store.ugm.d32.a64   [r24:4], [r24:4+0x4], [r24:4+0x8], [r24:4+0xC]
vecstore: 1× store.ugm.d32x4.a64 [r4:4]   r8:8
```

> **注**: 之前的分析错误地认为热路径使用 d16u32 store 并通过 byte-lane 拆分产生写放大。
> 实际热路径使用 d32 store（编译器已将 2 个 half 打包为 1 个 dword），
> 写放大来自 4 条独立 d32 message 各写 CL 的 1/4，而非 d16u32 的 byte-lane 拆分。
> 8 条 `d16u32` 仅出现在边界 fallback 路径（处理 start/remaining 不对齐的情况），
> 正常对齐数据不会走到。

#### Counter 变化总结

| 维度 | 变化 | 原因 |
|---|---|---|
| LSC WRITE_MESSAGE | **-75%** | 4 条 d32 store → 1 条 d32x4 store |
| LSC ACCESS | **-43%** | 4 条 store message 减少到 1 条，对应的 L1 cache access 减少 |
| LSC REGISTER_REQUEST | **-30~32%** | XVE 发出的 register request 减少（store message 减少） |
| L3_WRITE | **-75%** | 4 次 partial write 降至 1 次 full write |
| L3_STALL | **70% → 0%** | L3 不再因大量写请求而 stall |
| L3_SUPERQ_FULL | **12% → 0%** | L3 输入队列不再被写请求塞满 |
| L3_INPUT_AVAILABLE | **73% → 8%** | L3 请求压力大幅缓解 |
| L3_HIT | **-48~75%** | RFO 读入的 CL 被后续 partial write 命中（"假" hit），写减少后消失 |
| L3_MISS | **-32~38%** | RFO 触发的 L3 miss 消除（见下方分析） |
| GPU_MEMORY_BYTE_READ | **-43~50%** | Partial CL write 触发的 RFO DRAM 读消除（见下方分析） |
| GPU_MEMORY_BYTE_WRITE | 不变 | 实际写出数据量一致 |
| GPU_MEMORY_L3_READ | **-75~80%** | RFO 触发的 DRAM 读请求消除 |
| GpuTime | **-30~33%** | 整体加速 |

#### 为什么 4×d32 partial write 会导致额外 DRAM read

4 条 d32 store 最终写满同一个 CL，但问题在于**第一条 d32 store 到达时目标 CL 不在 L3 中**。

Softmax 的 output tensor 是新分配的或已被 evict 的内存区域。当第一条 d32 partial write 到达 L3 时：

1. 目标 CL 不在 L3 中（cold miss）
2. L3 不能凭空创建一个只有 1/4 字节有效的 CL（其余字节是垃圾数据）
3. L3 必须先从 DRAM **读回完整的 64B CL**（Read-For-Ownership / RFO），再合并 partial 数据
4. 后续 3 条 d32 store 写同一 CL 时，CL 已在 L3 中，不需要再 RFO

因此 **每个 output CL 产生恰好 1 次 RFO DRAM 读**。

**数据验证** — fp16 [1024, 32768]（output = 1,048,576 CLs）：

```
GPU_MEMORY_L3_READ:  main 1,310,822    vec 262,242
delta = 1,310,822 - 262,242 = 1,048,580 ≈ 1,048,576 = output CL 数量  ✓
```

main 和 vec 的差值精确等于 output CL 数量，说明每个 output CL 恰好被 RFO 了 1 次。

三个 shape 验证（delta / output_CLs）：

| Shape | main | vec | delta | output CLs | ratio |
|---|---|---|---|---|---|
| [1024,32768] | 1,310,822 | 262,242 | 1,048,580 | 1,048,576 | **1.0000** |
| [1024,65536] | 2,624,777 | 528,226 | 2,096,551 | 2,097,152 | **0.9997** |
| [1024,131072] | 5,512,754 | 1,319,306 | 4,193,448 | 4,194,304 | **0.9998** |

vec 版本本身的 `GPU_MEMORY_L3_READ`（如 262,242）是 softmax 正常 3-pass 读取 input 时的 L3 miss。Input 远大于 L3 容量（3.6~14.2x），跨行访问导致部分 L3 eviction，约 1/4 的 read 请求 miss 到 DRAM，这是正常的 capacity miss。

Vec store 版本用 1 条 d32x4 一次写满整个 CL（**full CL write**），L3 直接 allocate 新 CL 并填入完整数据，跳过 RFO，因此没有额外的 DRAM 读。

#### 完整因果链

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

#### 关键数量关系

每个 workitem 处理 vec_size=8 个 half output。SIMD32 下每条 store message 写 32 lanes：

| | main (4× d32) | vecstore (1× d32x4) |
|---|---|---|
| 每条 message 每 lane 写 | 4B (1 dword) | 16B (4 dwords) |
| 每条 message 总写 | 128B | 512B |
| 每 workitem 需要的 message 数 | 4 | 1 |
| 每 CL (64B) 被几条 message 写 | 4 (partial) | 1 (full) |
| 是否触发 RFO | 是（第一条 partial write） | 否（full write） |

### FP32: IGC 自动合并 — 无需 vec store

**ASM 确认**：FP32 main branch 的热路径已经是 `store.ugm.d32x4.a64`（1 条宽 store），
而不是 4× d32。IGC 对 float 的 4 次连续 d32 scalar store 成功做了 store merging。

```
FP32 main 热路径:   1× store.ugm.d32x4.a64 [r24:4]  r32:8    ← IGC 自动合并
FP32 vecstore 热路径: 1× store.ugm.d32x4.a64 [r16:4]  r20:8    ← 源码显式
```

两者生成的 store 指令完全一致，因此 counter 无差异。

#### 为什么 IGC 能合并 FP32 但不能合并 FP16？

关键在于 **计算结果到 store 之间的寄存器排布**：

- **FP32**: 计算在 float 上，结果直接是 `:f` 寄存器，4 个 float 自然排列在连续寄存器中。
  IGC 看到 4 条连续地址的 d32 store，源寄存器也连续 → 直接合并为 1 条 d32x4。

- **FP16**: 计算在 float 上（accscalar_t=float），结果需要 `float→half` 转换再写出。
  每个 half 结果只占 16-bit，需要通过 `mov :hf` + `mov :uw` 指令把 2 个 half
  打包进 1 个 dword 的高低 16-bit。这些 pack 指令导致寄存器排布不连续
  （stride-2 写入、交错的 uw move），打断了 IGC 的 store merging 窗口。

FP16 main 热路径 ASM 中 store 前的 pack 序列示例：
```asm
mov (32|M0)  r9.0<2>:hf    acc0.0<1;1,0>:f      // float→half, 写到偶数位
mov (32|M0)  r32.1<2>:uw   r9.0<2;1,0>:uw       // 把 half 搬到 dword 的高 16-bit
mov (32|M0)  r4.1<2>:uw    r9.0<2;1,0>:uw       // 类似的 pack 操作
...  // 多轮 pack 交错进行
store.ugm.d32.a64 (32|M0)  [r24:4]      r4:2    // 第 1 条 d32
store.ugm.d32.a64 (32|M0)  [r24:4+0x4]  r32:2   // 第 2 条 d32（寄存器不连续: r4 vs r32）
store.ugm.d32.a64 (32|M0)  [r24:4+0x8]  r12:2   // 第 3 条
store.ugm.d32.a64 (32|M0)  [r24:4+0xC]  r6:2    // 第 4 条（r4, r32, r12, r6 = 不连续）
```

注意 4 条 store 的源寄存器分别是 r4, r32, r12, r6 — **完全不连续**。
IGC 要合并为 d32x4 需要源数据在 8 个连续寄存器中（如 r8:8），
但 half→dword 的 pack 过程把数据分散到了不连续的寄存器。

vecstore patch 通过在源码中显式使用 `aligned_vector<half, 8>` store，
让编译器从一开始就把 8 个 half 结果排列到连续寄存器中，直接生成 d32x4。

#### 总结

| | FP16 main | FP16 vecstore | FP32 main | FP32 vecstore |
|---|---|---|---|---|
| 热路径 store | 4× d32 | **1× d32x4** | **1× d32x4** | 1× d32x4 |
| IGC 能 merge？ | 不能（pack 打断） | 源码显式 | 能（寄存器连续） | 源码显式 |
| partial write？ | 是 → RFO | 否 | 否 | 否 |
| 性能影响 | **-30%** | baseline | 无 | 无 |

**本质**：vec store patch 解决的是 IGC 对 FP16 store merging 的缺陷 —
half→dword pack 导致寄存器不连续，阻止了自动合并。
FP32 不需要这个 patch，因为 IGC 已经能自动合并。

---

## ASM 对比 (SoftmaxForwardKernelFunctor, fp16)

> 文件位置: `temp/softmax_asm/{main,vecstore}_fp16.asm`
> 编译方式: 从 SoftMaxKernels.cpp 提取 `SoftmaxForwardKernelFunctor`，单独编译:
> `icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2` + `IGC_ShaderDumpEnable=1`
> 模板参数: `<vec_size=8, half, half, float, uint32_t, LogSoftMax=false, is_safe_softmax=false>`

### 指令统计

| 指标 | main | vecstore | 差异 |
|---|---|---|---|
| ASM 行数 | 1718 | 1822 | +104 |
| `math.exp` | 24 | 24 | 相同 |
| `store.ugm.d32.a64` | 4 | 4 | 相同 (can_vec_store=false 路径) |
| `store.ugm.d16u32.a64` | 8 | 8 | 相同 (scalar fallback) |
| `store.ugm.d32x4.a64` | **0** | **1** | **新增 (can_vec_store=true 路径)** |

### 关键差异: 热路径 store 指令

**main branch** — `out_data_p[j] = result` 逐元素写:
```asm
// 编译器将 8 个 half 结果打包成 4 对 (2 half = 1 dword)，发出 4 条 d32 store
store.ugm.d32.a64 (32|M0)  [r24:4]      r4:2        // 写 2 个 half
store.ugm.d32.a64 (32|M0)  [r24:4+0x4]  r32:2       // 写 2 个 half
store.ugm.d32.a64 (32|M0)  [r24:4+0x8]  r12:2       // 写 2 个 half
store.ugm.d32.a64 (32|M0)  [r24:4+0xC]  r6:2        // 写 2 个 half
// 4 条 LSC store message，每条写 128 bytes (4B × 32 lanes)
// 总共 512 bytes = 正确覆盖 8 half × 32 lanes = 512 bytes
```

**vecstore branch** — `*(out_vec_t*)(...) = results` 向量写 (can_vec_store=true):
```asm
// 编译器将 8 个 half 打包到 r8:8 (8 个连续寄存器)，发出 1 条 d32x4 store
store.ugm.d32x4.a64 (32|M0)  [r4:4]  r8:8          // 一次写 8 个 half
// 1 条 LSC store message，写 512 bytes (16B × 32 lanes)
```

### 为什么 4×d32 → 1×d32x4 能带来 ~30% 性能提升？

1. **LSC message 数量减少 4×**: 每条 store message 需要经过 LSC→L1→L3 流水线，有固定的 descriptor decode / address generation / cache lookup 开销。4 条减少到 1 条。

2. **d32 store 写 half 数据 = partial write 问题**: 虽然编译器把 2 个 half 打包成 1 个 d32，但 4 条独立的 d32 store 写到同一个 cache line 时，每条只覆盖 CL 的 1/4。LSC 无法合并来自不同 message 的 partial writes → L3 看到 4 次 partial write → 第一次触发 RFO (read-for-ownership) → 额外 DRAM 读。

3. **d32x4 store = full cache line write**: 1 条 d32x4 message 每 lane 写 16 bytes，32 lanes × 16B = 512B。对于连续地址，这可以一次写满整个 64B cache line → L3 可以直接 allocate + fill，不需要 RFO → 消除额外 DRAM 读。

4. **Counter 数据验证**:
   - `XVE_LSC_WRITE_MESSAGE_COUNT`: main=524288, vec=131072 → **-75%** (4:1)
   - `LOAD_STORE_CACHE_ACCESS`: -42.8% (write message 减少导致)
   - `GPU_MEMORY_L3_READ`: -44% (RFO 消除)
   - `GPU_MEMORY_BYTE_READ`: -30~43% (DRAM 读减少)

### FP32 ASM 对比

FP32 (vec_size=4, float→float):
- main: 1193 行, vecstore: 1246 行
- **main 热路径已经是 `store.ugm.d32x4.a64`** — IGC 对 float 自动做了 store merging
- vecstore 热路径同样是 `store.ugm.d32x4.a64`（源码显式）
- 两者热路径 store 指令一致，因此 counter 无差异
- 4× d32 只出现在 boundary fallback 路径（两个分支都有）

# B60 vs L20: 大 dim_size 下 bandwidth efficiency 差距分析

> B60: MemBW 460GB/s. XeCore 20, L3 18MB, L1 256KB/XeCore.  Max WG 40. 0.5MB L3, 128KB L1 per WG

> L20:  MemBW 860GB/s. SM 92, L3 Cache 96MB, L1 128KB/SM.    Max CTA 92. 1MB L3, 128KB L1 per CTA,

### 性能对比数据（vec store 优化后）

| dtype | dim_size | io_bytes | B60 latency(us) | B60 BW(GB/s) | B60 bw_eff | L20 latency(us) | L20 BW(GB/s) | L20 bw_eff |
|---|---|---|---|---|---|---|---|---|
| fp16 | 32768 | 128MB | 357.28 | 375.67 | **81.7%** | 195.07 | 688.04 | **80.0%** |
| fp16 | 65536 | 256MB | 796.02 | 337.22 | **73.3%** | 396.80 | 676.50 | **78.7%** |
| fp16 | 131072 | 512MB | 1910.23 | 281.05 | **61.1%** | 790.63 | 679.04 | **79.0%** |
| fp32 | 32768 | 256MB | 707.70 | 379.31 | **82.5%** | 397.21 | 675.80 | **78.6%** |
| fp32 | 65536 | 512MB | 1838.55 | 292.01 | **63.5%** | 791.96 | 677.90 | **78.8%** |
| fp32 | 131072 | 1024MB | 4800.81 | 223.66 | **48.6%** | 2039.40 | 526.50 | **61.2%** |

**关键观察**: B60 在 dim_size=32768 时 bw_eff ~82%，与 L20 相当。但随着 dim_size 增大，B60 的 efficiency 急剧下降（82% → 61% → 49%），而 L20 保持在 ~79% 基本不变（131072/fp32 除外降到 61%）。

### 理论数据量 vs 实测数据量

Softmax 是 3-pass 算法（max, sum, div），理论 IO = read × 3 + write × 1。

| dtype | dim_size | 理论 IO | B60 theory | B60 real (profiler) | L20 theory | L20 real (profiler) |
|---|---|---|---|---|---|---|
| fp16 | 32768 | R 64MB×3, W 64MB | 40 WG × 64KB = 2.5MB R+W 同时 in-flight | Read L3 64MB, Write L3 64MB. Read Dram 64MB, Write Dram 75MB. L1 hit 69%, L3 hit 37% | R 6MB, W 6MB 同时 in-flight. I/O in L3 cache | Read L3 113MB, Write L3 67MB. Read Dram 74MB, Write Dram 21MB. L1 hit 32%, L3 hit 62% |
| fp16 | 65536 | R 128MB×3, W 128MB | 40 WG × 128KB = 5MB R+W 同时 in-flight.  | Read L3 223MB, Write L3 128MB. Read Dram 128MB, Write Dram 140MB. L1 hit 61%, L3 hit 70% | R 12MB, W 12MB 同时 in-flight. I/O in L3 cache | Read L3 393MB, Write L3 134MB. Read Dram 140MB, Write Dram 95MB. L1 hit 1.7%, L3 hit 74% |
| fp16 | 131072 | R 256MB×3, W 256MB | 40 WG × 256KB = 10MB R+W 同时 in-flight. L3 hold 不住 | Read L3 768MB, Write L3 256MB. Read Dram 332MB, Write Dram 264MB. L1 hit 55%, L3 hit 61% | R 24MB, W 24MB 同时 in-flight. I/O in L3 cache | Read L3 850MB, Write L3 268MB. Read Dram 300MB, Write Dram 254MB. L1 hit 0%, L3 hit 74% |
| fp32 | 32768 | R 128MB×3, W 128MB | 40 WG × 128KB = 5MB R+W 同时. | Read L3 244MB, Write L3 128MB. L1 hit 61%, L3 hit 60% | R 12MB, W 12MB 同时 in-flight. | Read L3 391MB, Write L3 134MB. Read Dram 150MB, Write Dram 93MB. L1 hit 2%, L3 hit 74% |
| fp32 | 65536 | R 256MB×3, W 256MB | 40 WG × 256KB = 10MB R+W 同时. L3 hold 不住 | Read L3 768MB, Write L3 256MB. L1 hit 56%, L3 hit 62% | R 24MB, W 24MB 同时 in-flight. | Read L3 800MB, Write L3 268MB. Read Dram 300MB, Write Dram 254MB. L1 hit 0%, L3 hit 75% |
| fp32 | 131072 | R 512MB×3, W 512MB | 40 WG × 512KB = 20MB R+W 同时. L3 hold 不住 | Read L3 1536MB, Write L3 512MB. Read Dram 1300MB, Write Dram 522MB. L1 hit 38%, L3 hit 40% | R 48MB, W 48MB 同时 in-flight. | Read L3 1.6GB, Write L3 536MB. Read Dram 671MB, Write Dram 564MB. L1 hit 0%, L3 hit 72% |

### 根因分析: B60 大 dim_size 下 efficiency 下降

核心问题是 **每个 workgroup/CTA 平均可用的 L3 cache 容量** 和 **3-pass 算法对 cache 复用的依赖**：

1. **每个 WG/CTA 平均可用 L3 容量**:
   - B60: L3 18MB / 40 WG = **460KB per WG**
   - L20: L2 96MB / 92 CTA = **1043KB per CTA**
   - L20 per-CTA 可用 cache 是 B60 的 **2.27x**

2. **Softmax 3-pass 数据复用模式**: Softmax 对每行 input 读 3 次（max pass, sum pass, div pass）。如果一行数据能留在 L3 cache 中，后两次读可以从 L3 命中，避免 DRAM 访问。每个 WG/CTA 处理一行，**行大小 vs per-WG 可用 cache 决定了复用效果**。

3. **per-WG 可用 cache 不足以 hold 一行数据时 efficiency 下降**:

   | dtype | dim_size | 每行大小 | B60 per-WG 可用 460KB | L20 per-CTA 可用 1043KB |
   |---|---|---|---|---|
   | fp16 | 32768 | 64KB | 64KB < 460KB ✓ hold 住 | 64KB < 1043KB ✓ hold 住 |
   | fp16 | 65536 | 128KB | 128KB < 460KB ✓ 但 3-pass 需 128KB×3=384KB，接近 460KB | 128KB < 1043KB ✓ 充裕 |
   | fp16 | 131072 | 256KB | 256KB×3=768KB > 460KB ✗ **hold 不住** | 256KB < 1043KB ✓ hold 住 |
   | fp32 | 32768 | 128KB | 128KB < 460KB ✓ | 128KB < 1043KB ✓ |
   | fp32 | 65536 | 256KB | 256KB×3=768KB > 460KB ✗ **hold 不住** | 256KB < 1043KB ✓ |
   | fp32 | 131072 | 512KB | 512KB×3=1536KB >> 460KB ✗ | 512KB×3=1536KB > 1043KB ✗ **hold 不住** |

   这与实测 bw_eff 的下降点精确吻合：
   - B60 fp16 从 dim=65536 开始掉（82% → 73%），dim=131072 进一步恶化（→ 61%）
   - B60 fp32 从 dim=65536 开始掉（82% → 64%），dim=131072 严重恶化（→ 49%）
   - L20 在 fp16/fp32 dim≤65536 时保持 ~79%，直到 fp32 dim=131072 才掉到 61%

4. **L3 miss 导致 DRAM 读放大**:
   - B60 fp16 dim=131072: 理论只需读 256MB 数据，但 Read L3 = 768MB（3x），说明 L3 无法缓存，每 pass 都 miss 到 DRAM。Read Dram = 332MB，加上 Write Dram 264MB = **596MB 实际 DRAM 流量**
   - L20 fp16 dim=131072: Read L3 = 850MB 类似，但 **L3 hit rate 74%** 远高于 B60 的 61%。Read Dram = 300MB + Write Dram 254MB = **554MB**

5. **fp32 dim=131072 时 L20 也掉效率**: 此时每行 512KB，3-pass 需 1536KB > per-CTA 可用 1043KB，L3 也 hold 不住。实测 L20 bw_eff 从 79% 降到 61%，Read Dram 飙升到 671MB。这进一步验证了 per-WG/CTA 可用 cache 是关键瓶颈。

#### 总结

| | B60 (L3 18MB, 40 WG, **460KB/WG**) | L20 (L2 96MB, 92 CTA, **1043KB/CTA**) |
|---|---|---|
| per-WG/CTA 可用 cache | 460KB | 1043KB (2.27x) |
| 一行数据能 hold 住的上限 (3-pass) | ~153KB/行 → dim ~78K (fp16), ~38K (fp32) | ~347KB/行 → dim ~178K (fp16), ~89K (fp32) |
| 实测 efficiency 开始下降的 dim | ~65536 (fp16), ~32768 (fp32) | ~131072 (fp32), fp16 未触及 |
| 超过上限后 | L3 hit rate 下降, DRAM 读放大, bw_eff 从 82% 降到 49% | L3 hit rate 保持 ~74%, bw_eff 保持 ~79% |
| 本质原因 | **per-WG 可用 L3 不足以缓存 3-pass 复用数据** | per-CTA 可用 L2 是 2.27x，能 hold 更大的行 |

---
