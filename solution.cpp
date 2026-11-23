/*
 * solution.cpp — YOUR SOLUTION.  Modify this file.
 *
 * Implement solve() to emit instructions that execute the compute graph
 * across multiple cores.  Your goal: minimise makespan (cycles).
 *
 * Available emit helpers (see hackathon.h for full docs):
 *
 *   emit_matmul(prog, core, in_addr, weight_addr, out_addr, M, K, N)
 *   emit_relu  (prog, core, in_addr, out_addr, size)
 *   emit_add   (prog, core, a_addr, b_addr, out_addr, size)
 *   emit_concat(prog, core, a_addr, b_addr, out_addr, len_a, len_b)
 *   emit_dma   (prog, core, src_core, src_addr, dst_addr, num_elements)
 *   emit_signal(prog, core, signal_id)
 *   emit_wait  (prog, core, signal_id)
 *
 * Cycle costs:
 *   matmul  = M * K * N
 *   relu    = size
 *   add     = size
 *   concat  = len_a + len_b
 *   dma     = num_elements * distance * prob->dma_factor (rounded);
 *             a copy inside the same core costs num_elements
 *   signal  = 0   (fires a named event)
 *   wait    = 0   (blocks until the event has fired)
 *
 * Rules:
 *   - Compute ops only read/write LOCAL sram.  Use DMA to pull remote data.
 *   - DMA reads from the source core's sram AS-IS at call time.
 *     Use signal/wait to ensure the source has finished writing first.
 *   - Final output must land on prob->output_core at prob->output_addr.
 *   - solve() must work for ANY problem: the final round uses hidden
 *     problems (other grids, graphs and placements). Test it with
 *     ./build/check --random 50
 *   - Each core has SRAM_FLOATS (4096) floats of local memory.
 *
 * Hints:
 *   - prob->ops[]             describes every graph operator
 *   - prob->initial_tensors[] tells you where each input tensor lives
 *   - manhattan_distance(a, b, prob->grid_cols) gives core-to-core distance
 *   - dma_cost(prob, dst_core, src_core, n) gives the cycles of a DMA
 *   - Row-splitting a matmul A[M x K] * B[K x N]:
 *       A_top = &A[0]            (M/2 * K floats, contiguous)
 *       A_bot = &A[(M/2)*K]      (M/2 * K floats, contiguous)
 *       C_top = A_top * B        -> M/2 * N floats
 *       C_bot = A_bot * B        -> M/2 * N floats
 *       Both halves use the SAME B matrix.
 */
#include "hackathon.h"

void solve(const Problem *prob, Program *prog)
{
    /*
     * TODO:  Replace this with your own scheduler.
     *
     * The starter code below is identical to the baseline:
     * everything on core 0, no parallelism.
     */
    const int core = 0;
    int next_addr = 0;
    int addr_of[MAX_GRAPH_OPS];
    std::memset(addr_of, -1, sizeof(addr_of));

    /* Pass 1: local tensors */
    for (int t = 0; t < prob->num_initial_tensors; t++) {
        const InitialTensor &it = prob->initial_tensors[t];
        if (it.core_id == core) {
            addr_of[it.op_id] = it.sram_addr;
            int end = it.sram_addr + it.num_floats;
            if (end > next_addr) next_addr = end;
        }
    }

    /* Pass 2: DMA remote tensors */
    for (int t = 0; t < prob->num_initial_tensors; t++) {
        const InitialTensor &it = prob->initial_tensors[t];
        if (it.core_id != core) {
            emit_dma(prog, core, it.core_id, it.sram_addr,
                     next_addr, it.num_floats);
            addr_of[it.op_id] = next_addr;
            next_addr += it.num_floats;
        }
    }

    /* Pass 3: execute graph ops */
    for (int i = 0; i < prob->num_ops; i++) {
        const GraphOp &op = prob->ops[i];
        if (op.type == GRAPH_INPUT) continue;

        int out_addr = next_addr;
        switch (op.type) {
        case GRAPH_MATMUL:
            emit_matmul(prog, core,
                        addr_of[op.input_ids[0]],
                        addr_of[op.input_ids[1]],
                        out_addr, op.M, op.K, op.N);
            next_addr += op.M * op.N;
            break;
        case GRAPH_RELU:
            emit_relu(prog, core, addr_of[op.input_ids[0]],
                      out_addr, op.output_size);
            next_addr += op.output_size;
            break;
        case GRAPH_ADD:
            emit_add(prog, core,
                     addr_of[op.input_ids[0]],
                     addr_of[op.input_ids[1]],
                     out_addr, op.output_size);
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

    /* Move the result to the output core and address */
    if (prob->output_core != core) {
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
