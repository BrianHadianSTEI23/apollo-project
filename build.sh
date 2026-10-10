#!/bin/bash
set -e

# Define target platform (adjust 202320_1 to match your platform version if different)
PLATFORM="/home/vitis-ai-user/workspace/kria-vitis-platforms/k26/platforms/xsct/k26_base_starter_kit/k26_base_starter_kit/export/k26_base_starter_kit/k26_base_starter_kit.xpfm"

echo "=== 1. Compiling HLS C++ Kernel to .xo ==="
v++ -c -t hw --platform $PLATFORM -k snn_kernel snn_kernel.cpp -o snn_kernel.xo

echo "=== 2. Linking .xo to generate .xclbin ==="
v++ -l -t hw --platform $PLATFORM snn_kernel.xo -o snn_kernel.xclbin

echo "=== 3. Extracting Raw Bitstream from .xclbin ==="
xclbinutil --input snn_kernel.xclbin --dump-section BITSTREAM:RAW:snn_app.bit

echo "=== 4. Converting .bit to .bit.bin using Bootgen ==="
echo "all: { [destination_device = pl] snn_app.bit }" > boot.bif
bootgen -image boot.bif -arch zynqmp -process_bitstream bin -w on

echo "=== 5. Compiling Device Tree Overlay (.dtbo) ==="
dtc -@ -I dts -O dtb -o snn_app.dtbo snn_app.dts

echo "=== 6. Cross-Compiling Host Executable for AArch64 ==="
aarch64-linux-gnu-g++ -O3 -std=c++17 main.cpp -o snn_host \
    -I$XILINX_XRT/include -L$XILINX_XRT/lib -lxrt_coreutil -pthread

echo "=== BUILD COMPLETE ==="