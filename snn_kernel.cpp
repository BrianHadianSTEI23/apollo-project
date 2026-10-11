#include <ap_int.h>

#define IN_SIZE 16
#define OUT_SIZE 10
#define TIMESTEPS 10
#define THRESHOLD 1.0f
#define DECAY 0.8f
#define LEARNING_RATE 0.05f

extern "C" {
void snn_kernel(
    const float* inputs,   // Input spikes
    float* weights,        // Synaptic weights
    float* outputs,        // Output spikes
    const float* targets,  // Target spike rates
    int mode               // 0 = Inference, 1 = Training
) {
    #pragma HLS INTERFACE m_axi port=inputs   offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=weights  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=outputs  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=targets  offset=slave bundle=gmem3

    #pragma HLS INTERFACE s_axilite port=inputs  bundle=control
    #pragma HLS INTERFACE s_axilite port=weights bundle=control
    #pragma HLS INTERFACE s_axilite port=outputs bundle=control
    #pragma HLS INTERFACE s_axilite port=targets bundle=control
    #pragma HLS INTERFACE s_axilite port=mode    bundle=control
    #pragma HLS INTERFACE s_axilite port=return  bundle=control

    // Local BRAM Buffers on FPGA
    float buf_inputs[TIMESTEPS * IN_SIZE];
    float buf_weights[IN_SIZE * OUT_SIZE];
    float buf_outputs[TIMESTEPS * OUT_SIZE];
    float buf_targets[OUT_SIZE];
    float v_mem[OUT_SIZE];

    #pragma HLS ARRAY_PARTITION variable=v_mem complete

    // --- 1. Burst Read from DDR RAM to Local BRAM ---
    read_inputs: for (int i = 0; i < TIMESTEPS * IN_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        buf_inputs[i] = inputs[i];
    }
    read_weights: for (int i = 0; i < IN_SIZE * OUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        buf_weights[i] = weights[i];
    }
    read_targets: for (int i = 0; i < OUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        buf_targets[i] = targets[i];
    }

    // Initialize membrane potentials
    for (int j = 0; j < OUT_SIZE; j++) {
        v_mem[j] = 0.0f;
    }

    // --- 2. Process SNN Logic strictly inside BRAM ---
    for (int t = 0; t < TIMESTEPS; t++) {
        for (int j = 0; j < OUT_SIZE; j++) {
            float input_current = 0.0f;
            for (int i = 0; i < IN_SIZE; i++) {
                #pragma HLS PIPELINE II=1
                input_current += buf_inputs[t * IN_SIZE + i] * buf_weights[i * OUT_SIZE + j];
            }

            v_mem[j] = (v_mem[j] * DECAY) + input_current;

            if (v_mem[j] >= THRESHOLD) {
                buf_outputs[t * OUT_SIZE + j] = 1.0f;
                v_mem[j] = 0.0f;
            } else {
                buf_outputs[t * OUT_SIZE + j] = 0.0f;
            }
        }
    }

    // --- 3. Training Weight Update in BRAM ---
    if (mode == 1) {
        for (int j = 0; j < OUT_SIZE; j++) {
            float total_out_spikes = 0.0f;
            for (int t = 0; t < TIMESTEPS; t++) {
                total_out_spikes += buf_outputs[t * OUT_SIZE + j];
            }
            
            float error = buf_targets[j] - (total_out_spikes / (float)TIMESTEPS);

            for (int i = 0; i < IN_SIZE; i++) {
                float total_in_spikes = 0.0f;
                for (int t = 0; t < TIMESTEPS; t++) {
                    total_in_spikes += buf_inputs[t * IN_SIZE + i];
                }
                buf_weights[i * OUT_SIZE + j] += LEARNING_RATE * error * total_in_spikes;
            }
        }
    }

    // --- 4. Burst Write from BRAM back to DDR RAM ---
    write_outputs: for (int i = 0; i < TIMESTEPS * OUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        outputs[i] = buf_outputs[i];
    }
    if (mode == 1) {
        write_weights: for (int i = 0; i < IN_SIZE * OUT_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            weights[i] = buf_weights[i];
        }
    }
}
}