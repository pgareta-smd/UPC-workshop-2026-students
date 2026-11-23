#ifndef HACKATHON_H
#define HACKATHON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/* ================================================================
 * Constants
 * ================================================================ */
#define MAX_CORES              16
#define SRAM_FLOATS            4096   /* per core: 4096 floats = 16 KB */
#define MAX_GRAPH_OPS          128
#define MAX_OP_INPUTS          4
#define MAX_INSTR_PER_CORE     512
#define MAX_SIGNALS            512
#define NUM_PROBLEMS           6

/* ================================================================
 * Compute-Graph Types  (problem definition — do not modify)
 * ================================================================ */
enum GraphOpType {
    GRAPH_INPUT,          /* source tensor, pre-loaded in SRAM           */
    GRAPH_MATMUL,         /* C[M×N] = A[M×K] · B[K×N]                   */
    GRAPH_RELU,           /* out[i] = max(0, in[i])                      */
    GRAPH_ADD,            /* out[i] = a[i] + b[i]                        */
    GRAPH_CONCAT          /* out = a || b  (flat concatenation)          */
};

struct GraphOp {
    int             id;
    GraphOpType     type;
    int             M, K, N;           /* MATMUL dims; unused otherwise  */
    int             output_size;       /* total floats in output tensor  */
    int             num_inputs;
    int             input_ids[MAX_OP_INPUTS];
};

struct InitialTensor {
    int     op_id;          /* which INPUT graph-op this tensor belongs to */
    int     core_id;        /* core whose SRAM holds the data             */
    int     sram_addr;      /* starting float-offset inside that core     */
    int     num_floats;
    float  *data;           /* pointer to the actual values               */
};

struct Problem {
    char            name[64];
    int             num_ops;
    GraphOp         ops[MAX_GRAPH_OPS];

    int             num_cores;
    int             grid_rows, grid_cols;
    double          dma_factor;        /* DMA cost multiplier (1.0 = normal) */

    int             num_initial_tensors;
    InitialTensor   initial_tensors[MAX_GRAPH_OPS];

    /* Where the final result must end up so the checker can verify it */
    int             output_op_id;
    int             output_core;
    int             output_addr;
    int             output_size;
    float          *expected_output;   /* filled by framework */
};

/* ================================================================
 * Instruction Set  (what students emit)
 * ================================================================ */
enum Opcode {
    OP_MATMUL,      /* matmul  on LOCAL sram                              */
    OP_RELU,        /* relu    on LOCAL sram                              */
    OP_ADD,         /* add     on LOCAL sram                              */
    OP_CONCAT,      /* concat  on LOCAL sram                              */
    OP_DMA,         /* copy from REMOTE core's sram to LOCAL sram         */
    OP_SIGNAL,      /* fire a named signal (zero-cost)                    */
    OP_WAIT         /* block until a named signal has been fired          */
};

struct Instruction {
    Opcode opcode;
    union {
        struct { int in_addr, weight_addr, out_addr, M, K, N; }  matmul;
        struct { int in_addr, out_addr, size; }                   relu;
        struct { int a_addr, b_addr, out_addr, size; }            add;
        struct { int a_addr, b_addr, out_addr, len_a, len_b; }    concat;
        struct { int src_core, src_addr, dst_addr, num_elements; } dma;
        struct { int id; }                                         signal;
        struct { int id; }                                         wait;
    };
};

struct Program {
    int          num_instr[MAX_CORES];
    Instruction  queues[MAX_CORES][MAX_INSTR_PER_CORE];
};

/* ================================================================
 * Simulation Result
 * ================================================================ */
struct SimResult {
    int     valid;                       /* 1 = schedule ran to completion */
    int     correct;                     /* 1 = output matches expected    */
    int     makespan;                    /* max core finish time (cycles)  */
    int     core_cycles[MAX_CORES];
    int     total_dma_cycles;
    int     total_compute_cycles;
    int     sync_warnings;               /* DMA reads without proper sync */
    char    error[256];
    float   sram[MAX_CORES][SRAM_FLOATS];
};

/* ================================================================
 * Emit Helpers — call these from solve() to build the program
 *
 *   All compute ops (matmul/relu/add/concat) operate ONLY on the
 *   executing core's local SRAM.  Use OP_DMA to pull remote data
 *   into local SRAM first.
 *
 *   Cycle costs:
 *     matmul  : M * K * N
 *     relu    : size
 *     add     : size
 *     concat  : len_a + len_b
 *     dma     : num_elements * distance * prob->dma_factor, rounded
 *               (local copy = num_elements * 1); see dma_cost()
 *     signal  : 0   (but records current time for waiters)
 *     wait    : 0   (but blocks until signal has fired; advances clock
 *                     to signal-fire time if that is later)
 * ================================================================ */
static inline void emit_matmul(Program *p, int core,
                               int in_addr, int weight_addr, int out_addr,
                               int M, int K, int N)
{
    int i = p->num_instr[core]++;
    Instruction &inst = p->queues[core][i];
    inst.opcode            = OP_MATMUL;
    inst.matmul.in_addr    = in_addr;
    inst.matmul.weight_addr= weight_addr;
    inst.matmul.out_addr   = out_addr;
    inst.matmul.M          = M;
    inst.matmul.K          = K;
    inst.matmul.N          = N;
}

static inline void emit_relu(Program *p, int core,
                             int in_addr, int out_addr, int size)
{
    int i = p->num_instr[core]++;
    Instruction &inst = p->queues[core][i];
    inst.opcode        = OP_RELU;
    inst.relu.in_addr  = in_addr;
    inst.relu.out_addr = out_addr;
    inst.relu.size     = size;
}

static inline void emit_add(Program *p, int core,
                            int a_addr, int b_addr, int out_addr, int size)
{
    int i = p->num_instr[core]++;
    Instruction &inst = p->queues[core][i];
    inst.opcode       = OP_ADD;
    inst.add.a_addr   = a_addr;
    inst.add.b_addr   = b_addr;
    inst.add.out_addr = out_addr;
    inst.add.size     = size;
}

static inline void emit_concat(Program *p, int core,
                               int a_addr, int b_addr, int out_addr,
                               int len_a, int len_b)
{
    int i = p->num_instr[core]++;
    Instruction &inst = p->queues[core][i];
    inst.opcode          = OP_CONCAT;
    inst.concat.a_addr   = a_addr;
    inst.concat.b_addr   = b_addr;
    inst.concat.out_addr = out_addr;
    inst.concat.len_a    = len_a;
    inst.concat.len_b    = len_b;
}

static inline void emit_dma(Program *p, int core,
                            int src_core, int src_addr,
                            int dst_addr, int num_elements)
{
    int i = p->num_instr[core]++;
    Instruction &inst = p->queues[core][i];
    inst.opcode             = OP_DMA;
    inst.dma.src_core       = src_core;
    inst.dma.src_addr       = src_addr;
    inst.dma.dst_addr       = dst_addr;
    inst.dma.num_elements   = num_elements;
}

static inline void emit_signal(Program *p, int core, int signal_id)
{
    int i = p->num_instr[core]++;
    Instruction &inst = p->queues[core][i];
    inst.opcode    = OP_SIGNAL;
    inst.signal.id = signal_id;
}

static inline void emit_wait(Program *p, int core, int signal_id)
{
    int i = p->num_instr[core]++;
    Instruction &inst = p->queues[core][i];
    inst.opcode  = OP_WAIT;
    inst.wait.id = signal_id;
}

/* ================================================================
 * Framework API  (implemented in hackathon.cpp)
 * ================================================================ */
int   manhattan_distance(int a, int b, int grid_cols);
/** Cycles of a DMA of @p num_elements floats from @p src_core to @p dst_core. */
int   dma_cost(const Problem *problem, int dst_core, int src_core, int num_elements);
void  load_problem(int test_id, Problem *problem);
int   simulate(const Problem *problem, const Program *program, SimResult *result);
int   verify_output(const Problem *problem, const SimResult *result);
void  print_problem(const Problem *problem);
void  print_result(const SimResult *result, int num_cores);

/* ================================================================
 * Student entry point  (implemented in solution.cpp / baseline.cpp)
 * ================================================================ */
void solve(const Problem *problem, Program *program);

#endif /* HACKATHON_H */
