#include <iostream>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <xclbin_path>" << std::endl;
        return 1;
    }

    std::string xclbin_path = argv[1];
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbin_path);
    auto kernel = xrt::kernel(device, uuid, "snn_kernel");

    std::cout << "[INFO] XRT Device and Kernel Loaded Successfully!" << std::endl;

    // Buffer sizes
    size_t in_size_bytes = 16 * 10 * sizeof(float);   // 16 inputs x 10 timesteps
    size_t weight_size_bytes = 16 * 10 * sizeof(float); // 16 inputs x 10 outputs
    size_t out_size_bytes = 10 * 10 * sizeof(float);  // 10 outputs x 10 timesteps
    size_t target_size_bytes = 10 * sizeof(float);

    // Allocate XRT Buffer Objects
    auto bo_in = xrt::bo(device, in_size_bytes, kernel.group_id(0));
    auto bo_weights = xrt::bo(device, weight_size_bytes, kernel.group_id(1));
    auto bo_out = xrt::bo(device, out_size_bytes, kernel.group_id(2));
    auto bo_targets = xrt::bo(device, target_size_bytes, kernel.group_id(3));

    // Map host pointers
    auto ptr_in = bo_in.map<float*>();
    auto ptr_weights = bo_weights.map<float*>();
    auto ptr_out = bo_out.map<float*>();
    auto ptr_targets = bo_targets.map<float*>();

    // Populate Dummy Data (Mode = 1: Training)
    for (int i = 0; i < 160; i++) ptr_in[i] = 1.0f;
    for (int i = 0; i < 160; i++) ptr_weights[i] = 0.1f;
    for (int i = 0; i < 10; i++) ptr_targets[i] = 0.5f;

    // Sync input buffers to PL DDR RAM
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_targets.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "[INFO] Triggering SNN Kernel Execution on PL..." << std::endl;

    // Run Kernel (mode = 1)
    auto run = kernel(bo_in, bo_weights, bo_out, bo_targets, 1);
    run.wait();

    // Sync output buffers back from PL DDR RAM to Host
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_weights.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    std::cout << "[SUCCESS] Kernel Execution Finished!" << std::endl;
    std::cout << "First Output Spike Value: " << ptr_out[0] << std::endl;
    std::cout << "Updated Weight[0] Value: " << ptr_weights[0] << std::endl;

    return 0;
}