# Intel Arc Pro B60 Graphics

## Device Identity
| Property | Value |
|----------|-------|
| name | Intel(R) Arc(TM) Pro B60 Graphics |
| vendor | Intel(R) Corporation |
| vendor_id | 32902 (0x8086) |
| ext_intel_device_id | 0xe211 |
| ext_intel_pci_address | 0000:29:00.0 |
| ext_intel_uuid | 868011e2-0000-0000-2900-000000000000 |
| device_type | gpu |
| driver_version | 1.14.37020+3 |
| Architecture | Xe2 / Battlemage (BMG-G21) |

## Platform
| Property | Value |
|----------|-------|
| platform_name | Intel(R) oneAPI Unified Runtime over Level-Zero V2 |
| platform_vendor | Intel(R) Corporation |
| platform_version | 1.14 |

## Compute Topology
| Property | Value |
|----------|-------|
| max_compute_units | 160 |
| ext_intel_gpu_eu_count | 160 |
| ext_intel_gpu_eu_simd_width | 16 |
| ext_intel_gpu_slices | 5 |
| ext_intel_gpu_subslices_per_slice | 4 |
| ext_intel_gpu_eu_count_per_subslice | 8 |
| ext_intel_gpu_hw_threads_per_eu | 8 |
| max_work_group_size | 1024 |
| max_work_item_dimensions | 3 |
| max_work_item_sizes | [1024, 1024, 1024] |
| max_num_sub_groups | 64 |
| sub_group_sizes | 16, 32 |

## Clock
| Property | Value |
|----------|-------|
| max_clock_frequency | 2183 MHz |

## Memory
| Property | Value |
|----------|-------|
| global_mem_size | 24385683456 bytes (22.71 GB) |
| global_mem_cache_size | 18874368 bytes (18 MB, L2) |
| global_mem_cache_line_size | 256 bytes |
| global_mem_cache_type | read_write |
| local_mem_size | 131072 bytes (128 KB) |
| ext_intel_memory_clock_rate | 2400 MHz |
| ext_intel_memory_bus_width | 64 bits (per-channel, NOT total aggregate) |
| Memory Type | GDDR6, 192-bit total bus, 19.2 Gbps effective |

## Peak Performance Projections

### Compute Throughput
| Precision | Path | Peak Throughput | Derivation |
|-----------|------|----------------|------------|
| FP32 | Vector Engine | **11.15 TFLOPS** | 160 EU × 16 ALU × 2(FMA) × 2183 MHz / 1000 |
| FP16 | Vector Engine | **22.29 TFLOPS** | FP32 VE × 2 (Xe2 packed-FP16) |
| FP16/BF16 | XMX Engine | **TBD** | Needs official spec confirmation |

### Memory Bandwidth
| Metric | Value |
|--------|-------|
| Peak BW | **460.8 GB/s** | 19.2 Gbps × 192-bit / 8 |

### Arithmetic Intensity Crossover (Ridge Point)
| Kernel Type | Hardware Path | Ridge Point (FLOP/Byte) |
|-------------|---------------|------------------------|
| Element-wise, reduction | VE FP32 | 24.2 |
| Element-wise, reduction | VE FP16 | 48.4 |

## Notes
- Architecture: Xe2/BMG (same die as Arc B580, BMG-G21)
- GPU clock is lower than consumer B580 (2183 vs 2850 MHz), but memory spec is similar
- `ext_intel_max_mem_bandwidth` not available from driver; BW calculated from GDDR6 spec
