cmake_minimum_required(VERSION 3.18)
project(xpu_device_prop LANGUAGES CXX)

# Build with: cmake -DCMAKE_CXX_COMPILER=icpx ..
add_library(xpu_device_prop SHARED xpuDeviceProp.cpp)
target_compile_options(xpu_device_prop PRIVATE -fsycl)
target_link_options(xpu_device_prop PRIVATE -fsycl)
set_target_properties(xpu_device_prop PROPERTIES
    LINKER_LANGUAGE CXX
    PREFIX ""
)
