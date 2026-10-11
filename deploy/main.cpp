#include <iostream>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

constexpr int IN_SIZE = 16;
constexpr int OUT_SIZE = 10;
constexpr int TIMESTEPS = 10;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <xclbin_file_path>" << std::endl;
        return 1;
    }

    std::string xclbin_path = argv[1];
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbin_path);
    auto snn_krnl = xrt::kernel(device, uuid, "snn_kernel");

    // Calculate buffer byte sizes
    size_t in_bytes  = TIMESTEPS * IN_SIZE * sizeof(float);
    size_t w_bytes   = IN_SIZE * OUT_SIZE * sizeof(float);
    size_t out_bytes = TIMESTEPS * OUT_SIZE * sizeof(float);
    size_t tgt_bytes = OUT_SIZE * sizeof(float);

    // Allocate XRT Buffer Objects (BOs)
    auto bo_inputs  = xrt::bo(device, in_bytes,  snn_krnl.group_id(0));
    auto bo_weights = xrt::bo(device, w_bytes,   snn_krnl.group_id(1));
    auto bo_outputs = xrt::bo(device, out_bytes, snn_krnl.group_id(2));
    auto bo_targets = xrt::bo(device, tgt_bytes, snn_krnl.group_id(3));

    // Map hardware buffers to host pointers
    float* ptr_inputs  = bo_inputs.map<float*>();
    float* ptr_weights = bo_weights.map<float*>();
    float* ptr_targets = bo_targets.map<float*>();

    // Initialize Host Input Spike Trains & Target Labels
    for (int i = 0; i < IN_SIZE * OUT_SIZE; ++i) ptr_weights[i] = 0.1f; 
    for (int i = 0; i < TIMESTEPS * IN_SIZE; ++i) ptr_inputs[i] = (i % 2 == 0) ? 1.0f : 0.0f;
    for (int i = 0; i < OUT_SIZE; ++i) ptr_targets[i] = (i == 0) ? 1.0f : 0.0f; 

    // Sync memory Host -> FPGA PL
    bo_inputs.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_targets.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "[INFO] Triggering On-Chip Training on KR260 PL..." << std::endl;
    auto run = snn_krnl(bo_inputs, bo_weights, bo_outputs, bo_targets, 1 /* Training Mode */);
    run.wait();

    // Read back updated synaptic weights from PL -> Host
    bo_weights.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::cout << "[SUCCESS] Training step complete. Updated weight [0][0]: " << ptr_weights[0] << std::endl;

    return 0;
}


