"""
Query XPU device properties via the compiled SYCL shared library.

Usage:
    cd projection/scripts
    python get_xpu_device_prop.py
"""

import ctypes
import os
import sys


def get_xpu_properties():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(script_dir, "build")

    search_dirs = [
        build_dir,
        os.path.join(build_dir, "Release"),
        os.path.join(build_dir, "Debug"),
    ]

    lib_path = None
    for d in search_dirs:
        path = os.path.join(d, "xpu_device_prop.so")
        if os.path.isfile(path):
            lib_path = path
            break

    if not lib_path:
        print("xpu_device_prop.so not found. Build first:")
        print("  cd projection/scripts && mkdir -p build && cd build")
        print("  cmake -DCMAKE_CXX_COMPILER=icpx .. && cmake --build . --config Release")
        sys.exit(1)

    lib = ctypes.CDLL(lib_path)
    lib.print_all_device_properties()


if __name__ == "__main__":
    get_xpu_properties()
