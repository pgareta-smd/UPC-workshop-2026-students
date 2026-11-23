/*
 * baseline.cpp — Naive single-core scheduler.
 *
 * Strategy: DMA all inputs to core 0, execute everything there
 * sequentially in topological order.  This is the score to beat.
 */
#include "hackathon.h"

void solve(const Problem *prob, Program *prog)
{
    const int core = 0;
    int next_addr = 0;

    /* addr_of[op_id] = SRAM address on core 0 where op's output lives */
    int addr_of[MAX_GRAPH_OPS];
    std::memset(addr_of, -1, sizeof(addr_of));

    /* Pass 1: register tensors already local to core 0.
     * This sets up next_addr so remote DMAs don't overwrite them. */
    for (int t = 0; t < prob->num_initial_tensors; t++) {
        const InitialTensor &it = prob->initial_tensors[t];
        if (it.core_id == core) {
            addr_of[it.op_id] = it.sram_addr;
            int end = it.sram_addr + it.num_floats;
            if (end > next_addr) next_addr = end;
        }
    }

    /* Pass 2: DMA remote tensors into core 0 after the local region */
    for (int t = 0; t < prob->num_initial_tensors; t++) {
        const InitialTensor &it = prob->initial_tensors[t];
        if (it.core_id != core) {
            int dst = next_addr;
            emit_dma(prog, core, it.core_id, it.sram_addr,
                     dst, it.num_floats);
            addr_of[it.op_id] = dst;
            next_addr += it.num_floats;
        }
    }

    /* Pass 3: execute graph ops in topological order */
    for (int i = 0; i < prob->num_ops; i++) {
        const GraphOp &op = prob->ops[i];
        if (op.type == GRAPH_INPUT) continue;

        int out_addr = next_addr;

        switch (op.type) {
        case GRAPH_MATMUL:
            emit_matmul(prog, core,
                        addr_of[op.input_ids[0]],
                        addr_of[op.input_ids[1]],
                        out_addr,
                        op.M, op.K, op.N);
            next_addr += op.M * op.N;
            break;

        case GRAPH_RELU:
            emit_relu(prog, core,
                      addr_of[op.input_ids[0]],
                      out_addr,
                      op.output_size);
            next_addr += op.output_size;
            break;

        case GRAPH_ADD:
            emit_add(prog, core,
                     addr_of[op.input_ids[0]],
                     addr_of[op.input_ids[1]],
                     out_addr,
                     op.output_size);
            next_addr += op.output_size;
            break;

        case GRAPH_CONCAT: {
            int la = prob->ops[op.input_ids[0]].output_size;
            int lb = prob->ops[op.input_ids[1]].output_size;
            emit_concat(prog, core,
                        addr_of[op.input_ids[0]],
                        addr_of[op.input_ids[1]],
                        out_addr, la, lb);
            next_addr += la + lb;
            break;
        }
        default: break;
        }

        addr_of[op.id] = out_addr;
    }

    /* Move result to required output address if needed */
    if (prob->output_core != core) {
        /* the output core pulls the result once core 0 is done */
        emit_signal(prog, core, 0);
        emit_wait(prog, prob->output_core, 0);
        emit_dma(prog, prob->output_core, core,
                 addr_of[prob->output_op_id],
                 prob->output_addr, prob->output_size);
    } else if (addr_of[prob->output_op_id] != prob->output_addr) {
        emit_dma(prog, core, core,
                 addr_of[prob->output_op_id],
                 prob->output_addr, prob->output_size);
    }
}
