# Intel Data Center GPU Max 1550

## Device Identity
| Property | Value |
|----------|-------|
| name | Intel(R) Data Center GPU Max 1550 |
| vendor | Intel(R) Corporation |
| vendor_id | 32902 (0x8086) |
| ext_intel_device_id | 0x0bd5 |
| ext_intel_pci_address | 0000:29:00.0 |
| ext_intel_uuid | 8680d50b-2f00-0000-2900-000000000001 |
| device_type | gpu |
| driver_version | 1.14.36711+4 |
| backend_version | 1.14.36711+4 |

## Platform
| Property | Value |
|----------|-------|
| platform_name | Intel(R) oneAPI Unified Runtime over Level-Zero |
| platform_vendor | Intel(R) Corporation |
| platform_version | 1.14 |

## Compute Topology
| Property | Value |
|----------|-------|
| max_compute_units | 512 |
| ext_intel_gpu_eu_count | 512 |
| ext_intel_gpu_eu_simd_width | 16 |
| ext_intel_gpu_slices | 1 |
| ext_intel_gpu_subslices_per_slice | 64 |
| ext_intel_gpu_eu_count_per_subslice | 8 |
| ext_intel_gpu_hw_threads_per_eu | 8 |
| max_work_group_size | 1024 |
| max_work_item_dimensions | 3 |
| max_work_item_sizes | [1024, 1024, 1024] |
| max_num_sub_groups | 64 |
| sub_group_sizes | 16, 32 |
| sub_group_independent_forward_progress | false |

## Clock
| Property | Value |
|----------|-------|
| max_clock_frequency | 1600 MHz |

## Memory
| Property | Value |
|----------|-------|
| global_mem_size | 68702699520 bytes (63.98 GB) |
| global_mem_cache_size | 201326592 bytes (192 MB) |
| global_mem_cache_line_size | 64 bytes |
| global_mem_cache_type | read_write |
| local_mem_size | 131072 bytes (128 KB) |
| mem_base_addr_align | 8 bits |
| ext_intel_free_memory | 68668022784 bytes |
| ext_intel_memory_clock_rate | 3200 MHz |
| ext_intel_memory_bus_width | 64 bits (per-channel, NOT total aggregate) |

## Power
| Property | Value |
|----------|-------|
| ext_intel_min_power_limit | -1 (not available) |
| ext_intel_max_power_limit | 600000 mW (600 W) |

## Queue
| Property | Value |
|----------|-------|
| ext_intel_max_compute_queue_indices | 1 |

## Atomics
| Property | Value |
|----------|-------|
| atomic_memory_order_capabilities | relaxed, acquire, release, acq_rel, seq_cst |
| atomic_fence_order_capabilities | relaxed, acquire, release, acq_rel, seq_cst |
| atomic_memory_scope_capabilities | work_item, sub_group, work_group, device, system |
| atomic_fence_scope_capabilities | work_item, sub_group, work_group, device, system |

## Vector Widths
| Type | Preferred | Native |
|------|-----------|--------|
| char | 16 | 16 |
| short | 8 | 8 |
| int | 4 | 4 |
| long | 1 | 1 |
| float | 1 | 1 |
| double | 1 | 1 |
| half | 8 | 8 |

## Derived Metrics
| Metric | Value |
|--------|-------|
| Peak Memory Bandwidth (GB/s) | 1600 GB/s per tile (3276.8 GB/s full card, 8× HBM2e stacks) |
| Peak Compute Throughput | See table below |
| Global Memory | 63.98 GB |
| Local Memory | 128 KB |

### Compute Throughput (per tile, 512 EUs @ 1600 MHz)
| Precision | Path | Peak Throughput | Source |
|-----------|------|----------------|--------|
| FP32 | Vector Engine | 26.2 TFLOPS | Official spec (52.4 TFLOPS full card / 2) |
| FP16/BF16 | XMX Engine | 419.4 TFLOPS | Official spec (838.86 TFLOPS full card / 2) |

### Roofline Crossover (per tile)
| Kernel Type | Hardware Path | Ridge Point (FLOP/Byte) |
|-------------|---------------|------------------------|
| Element-wise, reduction | VE FP32 | 16.4 |
| GEMM, convolution, attention | XMX BF16/FP16 | 262.1 |

## Notes
- `memory_bus_width` is per-channel (64 bits). For PVC Max 1550 with 8 HBM2e stacks, do NOT use this to calculate total bandwidth.
- `ext_intel_max_mem_bandwidth` is not available on this driver. Need user to provide.
- Peak FLOPS must be obtained from official architecture documentation, not derived from SYCL queries.
