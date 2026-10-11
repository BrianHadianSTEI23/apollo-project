#include <ap_int.h>

#define IN_SIZE 16
#define OUT_SIZE 10
#define TIMESTEPS 10
#define THRESHOLD 1.0f
#define DECAY 0.8f
#define LEARNING_RATE 0.05f

extern "C" {
void snn_kernel(
    const float* inputs,   // Input spikes: [TIMESTEPS x IN_SIZE]
    float* weights,        // Synaptic weights: [IN_SIZE x OUT_SIZE]
    float* outputs,        // Output spikes: [TIMESTEPS x OUT_SIZE]
    const float* targets,  // Target spike rates: [OUT_SIZE]
    int mode               // Mode selector: 0 = Inference, 1 = Training
) {
    // 1. AXI Master (m_axi) ports for global memory transfers
    #pragma HLS INTERFACE m_axi port=inputs   offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=weights  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=outputs  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=targets  offset=slave bundle=gmem3

    // 2. Map ALL pointer offset registers, scalars, and return into ONE s_axilite bundle ('control')
    #pragma HLS INTERFACE s_axilite port=inputs  bundle=control
    #pragma HLS INTERFACE s_axilite port=weights bundle=control
    #pragma HLS INTERFACE s_axilite port=outputs bundle=control
    #pragma HLS INTERFACE s_axilite port=targets bundle=control
    #pragma HLS INTERFACE s_axilite port=mode    bundle=control
    #pragma HLS INTERFACE s_axilite port=return  bundle=control

    float v_mem[OUT_SIZE];
    #pragma HLS ARRAY_PARTITION variable=v_mem complete

    // Initialize membrane potentials
    for (int j = 0; j < OUT_SIZE; j++) {
        #pragma HLS UNROLL
        v_mem[j] = 0.0f;
    }

    // --- Timestep Loop (Inference / Forward Pass) ---
    for (int t = 0; t < TIMESTEPS; t++) {
        #pragma HLS PIPELINE II=1
        
        for (int j = 0; j < OUT_SIZE; j++) {
            float input_current = 0.0f;
            
            for (int i = 0; i < IN_SIZE; i++) {
                input_current += inputs[t * IN_SIZE + i] * weights[i * OUT_SIZE + j];
            }

            // LIF Membrane Potential Dynamics
            v_mem[j] = (v_mem[j] * DECAY) + input_current;

            // Spike Generation and Reset
            if (v_mem[j] >= THRESHOLD) {
                outputs[t * OUT_SIZE + j] = 1.0f;
                v_mem[j] = 0.0f;
            } else {
                outputs[t * OUT_SIZE + j] = 0.0f;
            }
        }
    }

    // --- On-Chip Training Weight Update (Backward/Plasticity Pass) ---
    if (mode == 1) {
        for (int j = 0; j < OUT_SIZE; j++) {
            // Calculate output firing activity
            float total_out_spikes = 0.0f;
            for (int t = 0; t < TIMESTEPS; t++) {
                total_out_spikes += outputs[t * OUT_SIZE + j];
            }
            
            float error = targets[j] - (total_out_spikes / (float)TIMESTEPS);

            for (int i = 0; i < IN_SIZE; i++) {
                float total_in_spikes = 0.0f;
                for (int t = 0; t < TIMESTEPS; t++) {
                    total_in_spikes += inputs[t * IN_SIZE + i];
                }

                // Synaptic Plasticity Rule: Delta W = lr * Error * PreSpikes
                weights[i * OUT_SIZE + j] += LEARNING_RATE * error * total_in_spikes;
            }
        }
    }
}
}