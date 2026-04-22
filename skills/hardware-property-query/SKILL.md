# Hardware Property Query Skill

Query GPU device properties via SYCL/Level-Zero for Intel XPU devices.

> Compatible with: Claude Code, OpenCode, GitHub Copilot, Roo Code, and other AI coding assistants.

## Overview

This skill provides tools and guidance for querying hardware properties of Intel GPU devices, including:

1. Platform and driver information
2. Compute unit topology (EU count, slices, subslices, SIMD width)
3. Clock frequency
4. Memory properties (VRAM size, cache, local memory)
5. Intel extension properties (PCI ID, UUID, memory clock/bus, free memory, fan speed, power limits)
6. Atomic and vector width capabilities

## Directory Structure

```
hardware-property-query/
├── SKILL.md                      # This file
├── scripts/
│   ├── xpuDeviceProp.cpp         # SYCL device property query (C++ shared library source)
│   ├── CMakeLists.txt            # Build: cmake -DCMAKE_CXX_COMPILER=icpx ..
│   ├── get_xpu_device_prop.py    # Python entry point (loads .so via ctypes)
│   └── build/                    # Build artifacts: xpu_device_prop.so
```

## Build

```bash
source ~/intel/oneapi/setvars.sh   # Or the corresponding oneAPI environment
cd MySkills/skills/hardware-property-query/scripts
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=icpx ..
cmake --build . --config Release
```

## Run

```bash
cd MySkills/skills/hardware-property-query/scripts
python get_xpu_device_prop.py
```

The tool queries the **first GPU device on the Level-Zero platform** and outputs its complete properties.

## Output Properties

### Platform Info
- `platform_name`, `platform_vendor`, `platform_version`

### General Device Info
- `device_type`, `vendor_id`, `name`, `vendor`, `driver_version`, `version`, `backend_version`, `profile`

### Compute
- `max_compute_units` — total parallel compute units (EUs)
- `max_work_group_size`, `max_work_item_dimensions`, `max_work_item_sizes`
- `max_num_sub_groups`, `sub_group_sizes`, `sub_group_independent_forward_progress`

### Clock
- `max_clock_frequency` — max clock in MHz

### Memory
- `global_mem_size` — total VRAM in bytes
- `global_mem_cache_size`, `global_mem_cache_line_size`, `global_mem_cache_type`
- `local_mem_size`, `local_mem_type` — SLM per work-group
- `max_mem_alloc_size`, `mem_base_addr_align`

### Atomics
- `atomic_memory_order_capabilities`, `atomic_fence_order_capabilities`
- `atomic_memory_scope_capabilities`, `atomic_fence_scope_capabilities`

### Vector Widths
- `preferred_vector_width_*` and `native_vector_width_*` for char/short/int/long/float/double/half

### Intel Extension Properties (`ext::intel::info::device::*`)
- `ext_intel_device_id` — PCI device ID
- `ext_intel_pci_address` — BDF address
- `ext_intel_uuid` — device UUID
- `ext_intel_gpu_eu_count` — total Execution Units
- `ext_intel_gpu_eu_simd_width` — physical SIMD width per EU
- `ext_intel_gpu_slices` — GPU slices
- `ext_intel_gpu_subslices_per_slice`
- `ext_intel_gpu_eu_count_per_subslice`
- `ext_intel_gpu_hw_threads_per_eu`
- `ext_intel_max_mem_bandwidth` — peak memory bandwidth (bytes/s), if available
- `ext_intel_free_memory` — free VRAM (bytes)
- `ext_intel_memory_clock_rate` — memory clock in MHz
- `ext_intel_memory_bus_width` — **per-channel** bus width in bits (NOT total aggregate)
- `ext_intel_max_compute_queue_indices`
- `ext_intel_fan_speed`, `ext_intel_min_power_limit`, `ext_intel_max_power_limit`

### oneAPI Experimental Properties
- `ext_oneapi_max_global_work_groups`, `ext_oneapi_max_work_groups_3d`

### Derived Metrics
- **Peak Memory Bandwidth (GB/s)**: Only reported when `ext_intel_max_mem_bandwidth` is available. Not calculated from `memory_clock_rate` × `memory_bus_width` because the latter is per-channel width and unreliable for multi-channel HBM devices.
- **Global Memory (GB)**, **Local Memory (KB)**: Human-readable conversions.

## Known Limitations

1. **`memory_bus_width` is per-channel, not total**: For HBM devices (e.g., PVC Max 1550 with 8 HBM2e stacks), the driver reports a single channel's width (e.g., 64 bits), not the aggregate bus width. Do NOT use it to calculate total memory bandwidth.

3. **`ext_intel_max_mem_bandwidth` may not be available**: Some drivers/devices do not support this aspect. Ask user to provide info.

## Agent Workflow

When the user asks about GPU hardware properties:

1. Check if the device query tool is built (`scripts/build/libxpu_device_prop.so` or similar)
2. If not built, build it (see Build section above)
3. Run `python get_xpu_device_prop.py` to get live device properties
4. Present the relevant properties to the user
5. After querying device properties, generate a profile file under `hardware-projection/xpu_models/` for use by the hardware-projection skill: Collect live query results from step 3; Create the profile file with naming format: `<device_name>_<vendor_id>_<device_id>.md`
   - `device_name`: from `name`, lowercased with spaces replaced by `_` (e.g., `intel_data_center_gpu_max_1550`)
   - `vendor_id`: from `vendor_id` in hex (e.g., `8086`)
   - `device_id`: from `ext_intel_device_id` (e.g., `0bd5`)
   - Example: `intel_data_center_gpu_max_1550_8086_0bd5.md`
6. Save to: `MySkills/skills/hardware-projection/xpu_models/`