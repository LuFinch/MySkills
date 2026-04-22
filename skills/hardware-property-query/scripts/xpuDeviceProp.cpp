#include <sycl/sycl.hpp>
#include <stdio.h>
#include <vector>
#include <string>
#include <array>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// Suppress deprecation warnings: we intentionally query deprecated properties
// to provide a complete device property report.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

extern "C" EXPORT
void print_all_device_properties() {
    auto platforms = sycl::platform::get_platforms();
    int device_idx = 0;

    for (auto &platform : platforms) {
        // Only query Level-Zero platform
        auto backend = platform.get_backend();
        if (backend != sycl::backend::ext_oneapi_level_zero)
            continue;

        auto devices = platform.get_devices(sycl::info::device_type::gpu);
        for (auto &dev : devices) {
            // ----------------------------------------------------------------
            // Platform Info
            // ----------------------------------------------------------------
            std::string plat_name    = platform.get_info<sycl::info::platform::name>();
            std::string plat_vendor  = platform.get_info<sycl::info::platform::vendor>();
            std::string plat_version = platform.get_info<sycl::info::platform::version>();
            printf("  platform_name (platform name): %s\n",    plat_name.c_str());
            printf("  platform_vendor (platform vendor): %s\n", plat_vendor.c_str());
            printf("  platform_version (platform version): %s\n", plat_version.c_str());

            // ----------------------------------------------------------------
            // General Device Info
            // ----------------------------------------------------------------
            auto dev_type = dev.get_info<sycl::info::device::device_type>();
            const char* dev_type_str = "unknown";
            if      (dev_type == sycl::info::device_type::cpu)         dev_type_str = "cpu";
            else if (dev_type == sycl::info::device_type::gpu)         dev_type_str = "gpu";
            else if (dev_type == sycl::info::device_type::accelerator) dev_type_str = "accelerator";
            else if (dev_type == sycl::info::device_type::custom)      dev_type_str = "custom";
            else if (dev_type == sycl::info::device_type::automatic)   dev_type_str = "automatic";
            else if (dev_type == sycl::info::device_type::host)        dev_type_str = "host";
            printf("  device_type (device type): %s\n", dev_type_str);

            auto vendor_id = dev.get_info<sycl::info::device::vendor_id>();
            printf("  vendor_id (unique vendor ID): %u\n", vendor_id);

            std::string dev_name    = dev.get_info<sycl::info::device::name>();
            std::string dev_vendor  = dev.get_info<sycl::info::device::vendor>();
            std::string driver_ver  = dev.get_info<sycl::info::device::driver_version>();
            std::string backend_ver = dev.get_info<sycl::info::device::backend_version>();
            printf("  name (device name): %s\n",                         dev_name.c_str());
            printf("  vendor (device vendor): %s\n",                     dev_vendor.c_str());
            printf("  driver_version (driver version): %s\n",            driver_ver.c_str());
            printf("  backend_version (backend-specific version): %s\n", backend_ver.c_str());
            // ----------------------------------------------------------------
            // Compute
            // ----------------------------------------------------------------
            auto max_compute_units   = dev.get_info<sycl::info::device::max_compute_units>();
            auto max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();
            auto max_work_item_dims  = dev.get_info<sycl::info::device::max_work_item_dimensions>();
            auto max_work_item_sizes = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
            auto max_num_sub_groups  = dev.get_info<sycl::info::device::max_num_sub_groups>();
            auto sub_group_sizes     = dev.get_info<sycl::info::device::sub_group_sizes>();
            auto sub_group_ifp       = dev.get_info<sycl::info::device::sub_group_independent_forward_progress>();
            printf("  max_compute_units (max number of parallel compute units): %u\n", max_compute_units);
            printf("  max_work_group_size (max total work-items in a work-group): %zu\n", max_work_group_size);
            printf("  max_work_item_dimensions (max dimensions for work-items): %u\n", max_work_item_dims);
            printf("  max_work_item_sizes[0] (max work-items in dimension 0): %zu\n", max_work_item_sizes[0]);
            printf("  max_work_item_sizes[1] (max work-items in dimension 1): %zu\n", max_work_item_sizes[1]);
            printf("  max_work_item_sizes[2] (max work-items in dimension 2): %zu\n", max_work_item_sizes[2]);
            printf("  max_num_sub_groups (max number of sub-groups in a work-group): %u\n", max_num_sub_groups);
            printf("  sub_group_sizes (supported sub-group sizes):");
            for (auto sz : sub_group_sizes) printf(" %zu", sz);
            printf("\n");
            printf("  sub_group_independent_forward_progress (sub-groups make independent forward progress): %d\n", (int)sub_group_ifp);
            // ----------------------------------------------------------------
            // Clock
            // ----------------------------------------------------------------
            auto max_clock_frequency = dev.get_info<sycl::info::device::max_clock_frequency>();
            printf("  max_clock_frequency (max clock frequency in MHz): %u\n", max_clock_frequency);

            // ----------------------------------------------------------------
            // Memory
            // ----------------------------------------------------------------
            auto global_mem_size            = dev.get_info<sycl::info::device::global_mem_size>();
            auto global_mem_cache_size      = dev.get_info<sycl::info::device::global_mem_cache_size>();
            auto global_mem_cache_line_size = dev.get_info<sycl::info::device::global_mem_cache_line_size>();
            auto global_mem_cache_type      = dev.get_info<sycl::info::device::global_mem_cache_type>();
            printf("  global_mem_size (size of global memory in bytes): %llu\n", (unsigned long long)global_mem_size);
            printf("  global_mem_cache_size (size of global memory cache in bytes): %llu\n", (unsigned long long)global_mem_cache_size);
            printf("  global_mem_cache_line_size (global memory cache line size in bytes): %u\n", global_mem_cache_line_size);
            const char* cache_type_str = "none";
            if      (global_mem_cache_type == sycl::info::global_mem_cache_type::read_only)  cache_type_str = "read_only";
            else if (global_mem_cache_type == sycl::info::global_mem_cache_type::read_write) cache_type_str = "read_write";
            printf("  global_mem_cache_type (type of global memory cache): %s\n", cache_type_str);

            auto local_mem_size = dev.get_info<sycl::info::device::local_mem_size>();
            auto local_mem_type = dev.get_info<sycl::info::device::local_mem_type>();
            printf("  local_mem_size (size of local memory in bytes): %llu\n", (unsigned long long)local_mem_size);

            auto mem_base_addr_align   = dev.get_info<sycl::info::device::mem_base_addr_align>();
            printf("  mem_base_addr_align (minimum alignment for any data type in bits): %u\n", mem_base_addr_align);

            // ----------------------------------------------------------------
            // Atomics
            // ----------------------------------------------------------------
            auto print_memory_orders = [](const char* label, const char* desc, auto caps) {
                printf("  %s (%s):", label, desc);
                for (auto cap : caps) {
                    if      (cap == sycl::memory_order::relaxed) printf(" relaxed");
                    else if (cap == sycl::memory_order::acquire) printf(" acquire");
                    else if (cap == sycl::memory_order::release) printf(" release");
                    else if (cap == sycl::memory_order::acq_rel) printf(" acq_rel");
                    else if (cap == sycl::memory_order::seq_cst) printf(" seq_cst");
                }
                printf("\n");
            };
            auto print_memory_scopes = [](const char* label, const char* desc, auto caps) {
                printf("  %s (%s):", label, desc);
                for (auto cap : caps) {
                    if      (cap == sycl::memory_scope::work_item)  printf(" work_item");
                    else if (cap == sycl::memory_scope::sub_group)  printf(" sub_group");
                    else if (cap == sycl::memory_scope::work_group) printf(" work_group");
                    else if (cap == sycl::memory_scope::device)     printf(" device");
                    else if (cap == sycl::memory_scope::system)     printf(" system");
                }
                printf("\n");
            };
            print_memory_orders("atomic_memory_order_capabilities", "supported memory orders for atomic operations",       dev.get_info<sycl::info::device::atomic_memory_order_capabilities>());
            print_memory_orders("atomic_fence_order_capabilities",  "supported memory orders for atomic fence operations",  dev.get_info<sycl::info::device::atomic_fence_order_capabilities>());
            print_memory_scopes("atomic_memory_scope_capabilities", "supported memory scopes for atomic operations",        dev.get_info<sycl::info::device::atomic_memory_scope_capabilities>());
            print_memory_scopes("atomic_fence_scope_capabilities",  "supported memory scopes for atomic fence operations",  dev.get_info<sycl::info::device::atomic_fence_scope_capabilities>());

            // ----------------------------------------------------------------
            // Preferred / Native Vector Widths
            // ----------------------------------------------------------------
            printf("  preferred_vector_width_char (preferred native vector width for char): %u\n",    dev.get_info<sycl::info::device::preferred_vector_width_char>());
            printf("  preferred_vector_width_short (preferred native vector width for short): %u\n",  dev.get_info<sycl::info::device::preferred_vector_width_short>());
            printf("  preferred_vector_width_int (preferred native vector width for int): %u\n",      dev.get_info<sycl::info::device::preferred_vector_width_int>());
            printf("  preferred_vector_width_long (preferred native vector width for long): %u\n",    dev.get_info<sycl::info::device::preferred_vector_width_long>());
            printf("  preferred_vector_width_float (preferred native vector width for float): %u\n",  dev.get_info<sycl::info::device::preferred_vector_width_float>());
            printf("  preferred_vector_width_double (preferred native vector width for double): %u\n",dev.get_info<sycl::info::device::preferred_vector_width_double>());
            printf("  preferred_vector_width_half (preferred native vector width for half): %u\n",    dev.get_info<sycl::info::device::preferred_vector_width_half>());
            printf("  native_vector_width_char (native ISA vector width for char): %u\n",             dev.get_info<sycl::info::device::native_vector_width_char>());
            printf("  native_vector_width_short (native ISA vector width for short): %u\n",           dev.get_info<sycl::info::device::native_vector_width_short>());
            printf("  native_vector_width_int (native ISA vector width for int): %u\n",               dev.get_info<sycl::info::device::native_vector_width_int>());
            printf("  native_vector_width_long (native ISA vector width for long): %u\n",             dev.get_info<sycl::info::device::native_vector_width_long>());
            printf("  native_vector_width_float (native ISA vector width for float): %u\n",           dev.get_info<sycl::info::device::native_vector_width_float>());
            printf("  native_vector_width_double (native ISA vector width for double): %u\n",         dev.get_info<sycl::info::device::native_vector_width_double>());
            printf("  native_vector_width_half (native ISA vector width for half): %u\n",             dev.get_info<sycl::info::device::native_vector_width_half>());

            // ----------------------------------------------------------------
            // Intel Extension Properties (ext::intel::info::device::*)
            // ----------------------------------------------------------------
            printf("  -- Intel Extension Properties --\n");

            if (dev.has(sycl::aspect::ext_intel_device_id)) {
                printf("  ext_intel_device_id (PCI device ID): 0x%x\n",
                       dev.get_info<sycl::ext::intel::info::device::device_id>());
            }
            if (dev.has(sycl::aspect::ext_intel_pci_address)) {
                auto pci_addr = dev.get_info<sycl::ext::intel::info::device::pci_address>();
                printf("  ext_intel_pci_address (PCI bus:device.function address): %s\n", pci_addr.c_str());
            }
            if (dev.has(sycl::aspect::ext_intel_device_info_uuid)) {
                auto uuid = dev.get_info<sycl::ext::intel::info::device::uuid>();
                printf("  ext_intel_uuid (device universally unique identifier):");
                for (int i = 0; i < (int)uuid.size(); i++) {
                    if (i == 4 || i == 6 || i == 8 || i == 10) printf("-");
                    printf("%02x", (unsigned)uuid[i]);
                }
                printf("\n");
            }
            if (dev.has(sycl::aspect::ext_intel_gpu_eu_count)) {
                printf("  ext_intel_gpu_eu_count (total number of Execution Units): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>());
            }
            if (dev.has(sycl::aspect::ext_intel_gpu_eu_simd_width)) {
                printf("  ext_intel_gpu_eu_simd_width (physical SIMD width of an EU): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>());
            }
            if (dev.has(sycl::aspect::ext_intel_gpu_slices)) {
                printf("  ext_intel_gpu_slices (number of slices in the GPU): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::gpu_slices>());
            }
            if (dev.has(sycl::aspect::ext_intel_gpu_subslices_per_slice)) {
                printf("  ext_intel_gpu_subslices_per_slice (number of subslices per slice): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>());
            }
            if (dev.has(sycl::aspect::ext_intel_gpu_eu_count_per_subslice)) {
                printf("  ext_intel_gpu_eu_count_per_subslice (number of EUs per subslice): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>());
            }
            if (dev.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) {
                printf("  ext_intel_gpu_hw_threads_per_eu (number of hardware threads per EU): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>());
            }
            if (dev.has(sycl::aspect::ext_intel_max_mem_bandwidth)) {
                printf("  ext_intel_max_mem_bandwidth (max memory bandwidth in bytes/second): %llu\n",
                       (unsigned long long)dev.get_info<sycl::ext::intel::info::device::max_mem_bandwidth>());
            }
            if (dev.has(sycl::aspect::ext_intel_free_memory)) {
                printf("  ext_intel_free_memory (free global memory in bytes): %llu\n",
                       (unsigned long long)dev.get_info<sycl::ext::intel::info::device::free_memory>());
            }
            if (dev.has(sycl::aspect::ext_intel_memory_clock_rate)) {
                printf("  ext_intel_memory_clock_rate (memory clock rate in MHz): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>());
            }
            if (dev.has(sycl::aspect::ext_intel_memory_bus_width)) {
                printf("  ext_intel_memory_bus_width (memory bus width in bits): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::memory_bus_width>());
            }
            try {
                auto idx = dev.get_info<sycl::ext::intel::info::device::max_compute_queue_indices>();
                printf("  ext_intel_max_compute_queue_indices (max queue indices for ordered queues): %d\n", idx);
            } catch (...) {}

            if (dev.has(sycl::aspect::ext_intel_device_info_luid)) {
                auto luid = dev.get_info<sycl::ext::intel::info::device::luid>();
                printf("  ext_intel_luid (locally unique identifier):");
                for (auto b : luid) printf(" %02x", (unsigned)b);
                printf("\n");
                printf("  ext_intel_node_mask (node mask for the device): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::node_mask>());
            }
            if (dev.has(sycl::aspect::ext_intel_fan_speed)) {
                printf("  ext_intel_fan_speed (current fan speed in RPM, -1 if not spinning): %d\n",
                       dev.get_info<sycl::ext::intel::info::device::fan_speed>());
            }
            if (dev.has(sycl::aspect::ext_intel_power_limits)) {
                printf("  ext_intel_min_power_limit (minimum power limit in milliwatts): %d\n",
                       dev.get_info<sycl::ext::intel::info::device::min_power_limit>());
                printf("  ext_intel_max_power_limit (maximum power limit in milliwatts): %d\n",
                       dev.get_info<sycl::ext::intel::info::device::max_power_limit>());
            }

            // ----------------------------------------------------------------
            // Derived Metrics
            // NOTE: Only Memory Bandwidth is calculated here from driver-
            // reported values. Compute throughput (FP32/FP16/BF16/INT8,
            // both Vector Engine and XMX) CANNOT be reliably derived from
            // SYCL queries — SIMD_width does not necessarily equal actual
            // ALU count per EU. Consult official device specs or ask the
            // user for architecture-specific ALU information.
            // ----------------------------------------------------------------
            printf("  -- Derived Metrics --\n");

            // Peak Memory Bandwidth
            if (dev.has(sycl::aspect::ext_intel_max_mem_bandwidth)) {
                auto max_bw = dev.get_info<sycl::ext::intel::info::device::max_mem_bandwidth>();
                printf("  Peak Memory Bandwidth (GB/s): %.2f\n", (double)max_bw / 1.0e9);
            } else {
                // NOTE: ext_intel_memory_bus_width reports per-channel width (e.g. 64 bits),
                // not total bus width, so it cannot be used to derive peak bandwidth for
                // multi-channel HBM devices. Use clpeak or official specs instead.
                printf("  Peak Memory Bandwidth (GB/s): n/a (ext_intel_max_mem_bandwidth not available; use official specs)\n");
            }

            printf("  Peak Compute Throughput: n/a (not calculated — SYCL-reported SIMD_width may not reflect actual ALU count; consult official device specs)\n");

            // Global Memory Size (human-readable)
            printf("  Global Memory (GB): %.2f\n", (double)global_mem_size / (1024.0 * 1024.0 * 1024.0));
            printf("  Local Memory (KB): %.2f\n",  (double)local_mem_size / 1024.0);

            printf("\n");
            device_idx++;
            break;
        }
        if (device_idx > 0) break;
    }

    if (device_idx == 0) {
        printf("No XPU/GPU devices found via SYCL.\n");
    } else {
        printf("Total XPU devices: %d\n", device_idx);
    }
}

#pragma clang diagnostic pop
