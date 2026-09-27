#pragma once
#include "whisper_types.h"
#include "hls_stream.h"

// Union to copy raw bits into fp16 without the C++ compiler attempting any arithmetic conversion
union HalfInt {
    uint16_t i;
    half f; 
    HalfInt() : i(0) {}
};

void hbm_to_q5_1_stream(
    const bus_t* weights_in_hbm,      // AXI4 pointer to HBM memory
    int total_blocks,                 // How many blocks of 32 weights we need
    hls::stream<block_q5_1>& stream_out // The output pipe towards our multiplier
) {
    // Our giant "gearbox" register
    ap_uint<1024> buffer = 0; 
    int bits_in_buffer = 0;
    int read_idx = 0;

    gearbox_loop: for (int b = 0; b < total_blocks; b++) {
        // We want to extract 1 block EXACTLY EVERY CLOCK CYCLE
        #pragma HLS PIPELINE II=1
        
        // 1. Refill the buffer if we run out of bits
        if (bits_in_buffer < 192) {
            // We read 512 bits from HBM and place them in the upper part of the buffer
            buffer = buffer | ((ap_uint<1024>)(weights_in_hbm[read_idx++]) << bits_in_buffer);
            bits_in_buffer += 512;
        }

        // 2. Extract exactly the lowest 192 bits
        ap_uint<192> raw_block = buffer(191, 0);
        
        // 3. Shift the buffer to discard what we already read
        buffer = buffer >> 192;
        bits_in_buffer -= 192;

        // --- HARDWARE MAGIC: BIT MAPPING ---
        block_q5_1 parsed_block;
        HalfInt h_caster;
        
        // Extract Scale (d) - Bits 0 to 15
        h_caster.i = raw_block(15, 0);
        parsed_block.d = h_caster.f;

        // Extract Minimum (m) - Bits 16 to 31
        h_caster.i = raw_block(31, 16);
        parsed_block.m = h_caster.f;

        // Extract qh (The fifth bit) - Bits 32 to 63
        parsed_block.qh = raw_block(63, 32);

        // Extract qs (The lower 16 bytes) - Bits 64 to 191
        ap_uint<128> qs_raw = raw_block(191, 64);
        
        // In hardware this loop does not exist, it is just wires split 8 bits at a time
        for (int i = 0; i < 16; i++) {
            #pragma HLS UNROLL
            parsed_block.qs[i] = qs_raw( (i * 8) + 7, i * 8 );
        }

        // Send to the multiplier
        stream_out.write(parsed_block);
    }
}