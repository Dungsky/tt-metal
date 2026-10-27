// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Horizon fabric unicast write — checker side (runs on chip 1).
//
// 1. Spins on recv_l1_addr[0] until non-zero (the sender's first word arrives).
// 2. Verifies all num_words words match base_value+i.
// 3. Writes 1u (pass) or 0xDEAD0000|i (fail at word i) to result_l1_addr.
// 4. Exits; host calls Finish(chip1) which waits for this point.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t recv_l1_addr = get_arg(args::recv_l1_addr);
    const uint32_t num_words = get_arg(args::num_words);
    const uint32_t base_value = get_arg(args::base_value);
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);

    volatile uint32_t* recv = reinterpret_cast<volatile uint32_t*>(recv_l1_addr);
    volatile uint32_t* result = reinterpret_cast<volatile uint32_t*>(result_l1_addr);

    // Spin until the sender's first word arrives (non-zero sentinel).
    while (recv[0] == 0) { /* spin */
    }

    // Verify all words.
    uint32_t pass = 1u;
    for (uint32_t i = 0; i < num_words; i++) {
        if (recv[i] != base_value + i) {
            pass = 0xDEAD0000u | i;
            break;
        }
    }

    *result = pass;
}
