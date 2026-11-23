/*
 * hackathon.cpp — Simulator, reference executor, test problems, main.
 * DO NOT MODIFY THIS FILE.
 */
#include "hackathon.h"
#include <cassert>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <chrono>
#include <cstdarg>

/* ================================================================
 * Utilities
 * ================================================================ */
int manhattan_distance(int a, int b, int grid_cols)
{
    int ra = a / grid_cols, ca = a % grid_cols;
    int rb = b / grid_cols, cb = b % grid_cols;
    return std::abs(ra - rb) + std::abs(ca - cb);
}

int dma_cost(const Problem *prob, int dst_core, int src_core, int num_elements)
{
    int dist = manhattan_distance(dst_core, src_core, prob->grid_cols);
    if (dist == 0) return num_elements;   /* local copy: 1 cycle/elem */
    double f = prob->dma_factor > 0 ? prob->dma_factor : 1.0;
    long cost = std::lround((double)num_elements * dist * f);
    return cost < 1 ? 1 : (int)cost;
}

static float *alloc_floats(int n)
{
    float *p = (float *)std::calloc(n, sizeof(float));
    if (!p) { std::fprintf(stderr, "out of memory\n"); std::exit(1); }
    return p;
}

/* deterministic data fill seeded by `seed` */
static float *make_data(int size, int seed)
{
    float *d = alloc_floats(size);
    unsigned s = (unsigned)seed;
    for (int i = 0; i < size; i++) {
        s = s * 1103515245u + 12345u;
        d[i] = (float)((s >> 16) & 0xFF) / 255.0f + 0.01f;
    }
    return d;
}

/* ================================================================
 * Reference compute kernels (golden model)
 * ================================================================ */
static void ref_matmul(const float *A, const float *B, float *C,
                       int M, int K, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int kk = 0; kk < K; kk++)
                sum += A[i * K + kk] * B[kk * N + j];
            C[i * N + j] = sum;
        }
}

static void ref_relu(const float *in, float *out, int size)
{
    for (int i = 0; i < size; i++)
        out[i] = in[i] > 0 ? in[i] : 0;
}

static void ref_add(const float *a, const float *b, float *out, int size)
{
    for (int i = 0; i < size; i++)
        out[i] = a[i] + b[i];
}

static void ref_concat(const float *a, const float *b, float *out,
                       int len_a, int len_b)
{
    std::memcpy(out, a, len_a * sizeof(float));
    std::memcpy(out + len_a, b, len_b * sizeof(float));
}

/* ================================================================
 * Reference graph execution  (single-threaded golden output)
 * ================================================================ */
static void compute_expected_output(Problem *p)
{
    float *outputs[MAX_GRAPH_OPS];
    std::memset(outputs, 0, sizeof(outputs));

    /* seed INPUT ops from initial tensors */
    for (int t = 0; t < p->num_initial_tensors; t++) {
        InitialTensor *it = &p->initial_tensors[t];
        outputs[it->op_id] = alloc_floats(it->num_floats);
        std::memcpy(outputs[it->op_id], it->data, it->num_floats * sizeof(float));
    }

    /* topological-order execute (ids are already in topo order) */
    for (int i = 0; i < p->num_ops; i++) {
        GraphOp *op = &p->ops[i];
        if (op->type == GRAPH_INPUT) continue;

        outputs[op->id] = alloc_floats(op->output_size);

        switch (op->type) {
        case GRAPH_MATMUL:
            ref_matmul(outputs[op->input_ids[0]],
                       outputs[op->input_ids[1]],
                       outputs[op->id],
                       op->M, op->K, op->N);
            break;
        case GRAPH_RELU:
            ref_relu(outputs[op->input_ids[0]],
                     outputs[op->id],
                     op->output_size);
            break;
        case GRAPH_ADD:
            ref_add(outputs[op->input_ids[0]],
                    outputs[op->input_ids[1]],
                    outputs[op->id],
                    op->output_size);
            break;
        case GRAPH_CONCAT: {
            int la = p->ops[op->input_ids[0]].output_size;
            int lb = p->ops[op->input_ids[1]].output_size;
            ref_concat(outputs[op->input_ids[0]],
                       outputs[op->input_ids[1]],
                       outputs[op->id], la, lb);
            break;
        }
        default: break;
        }
    }

    p->expected_output = alloc_floats(p->output_size);
    std::memcpy(p->expected_output,
                outputs[p->output_op_id],
                p->output_size * sizeof(float));

    for (int i = 0; i < p->num_ops; i++)
        std::free(outputs[i]);
}

/* ================================================================
 * Problem Definitions
 * ================================================================
 *
 * Core layout (2x2):          Core layout (3x3):
 *   C0 -- C1                    C0 -- C1 -- C2
 *   |     |                     |     |     |
 *   C2 -- C3                    C3 -- C4 -- C5
 *                                |     |     |
 *                                C6 -- C7 -- C8
 * ================================================================ */

/* helper: add a graph op */
static void set_input_op(Problem *p, int id, int out_size)
{
    GraphOp &o  = p->ops[id];
    std::memset(&o, 0, sizeof(o));
    o.id          = id;
    o.type        = GRAPH_INPUT;
    o.output_size = out_size;
}

static void set_matmul_op(Problem *p, int id, int a, int b,
                          int M, int K, int N)
{
    GraphOp &o    = p->ops[id];
    std::memset(&o, 0, sizeof(o));
    o.id            = id;
    o.type          = GRAPH_MATMUL;
    o.M = M; o.K = K; o.N = N;
    o.output_size   = M * N;
    o.num_inputs    = 2;
    o.input_ids[0]  = a;
    o.input_ids[1]  = b;
}

static void set_relu_op(Problem *p, int id, int in_id, int sz)
{
    GraphOp &o    = p->ops[id];
    std::memset(&o, 0, sizeof(o));
    o.id            = id;
    o.type          = GRAPH_RELU;
    o.output_size   = sz;
    o.num_inputs    = 1;
    o.input_ids[0]  = in_id;
}

static void set_add_op(Problem *p, int id, int a, int b, int sz)
{
    GraphOp &o    = p->ops[id];
    std::memset(&o, 0, sizeof(o));
    o.id            = id;
    o.type          = GRAPH_ADD;
    o.output_size   = sz;
    o.num_inputs    = 2;
    o.input_ids[0]  = a;
    o.input_ids[1]  = b;
}

static void set_concat_op(Problem *p, int id, int a, int b, int sz)
{
    GraphOp &o    = p->ops[id];
    std::memset(&o, 0, sizeof(o));
    o.id            = id;
    o.type          = GRAPH_CONCAT;
    o.output_size   = sz;
    o.num_inputs    = 2;
    o.input_ids[0]  = a;
    o.input_ids[1]  = b;
}

static void place_tensor(Problem *p, int idx, int op_id,
                         int core, int addr, int nfloats, int seed)
{
    InitialTensor &t = p->initial_tensors[idx];
    t.op_id      = op_id;
    t.core_id    = core;
    t.sram_addr  = addr;
    t.num_floats = nfloats;
    t.data       = make_data(nfloats, seed);
}

/* -------------------------------------------------------------- */
/* P0: single matmul  (warmup)                                     */
/*     A[4x4] @ C0,  B[4x4] @ C1  ->  C = A*B  -> output @ C0:0  */
/* -------------------------------------------------------------- */
static void load_p0(Problem *p)
{
    std::strcpy(p->name, "0_single_matmul");
    p->num_cores = 4;  p->grid_rows = 2;  p->grid_cols = 2;

    p->num_ops = 3;
    set_input_op(p, 0, 16);                    /* A  4x4 */
    set_input_op(p, 1, 16);                    /* B  4x4 */
    set_matmul_op(p, 2, 0, 1, 4, 4, 4);       /* C = A*B */

    p->num_initial_tensors = 2;
    place_tensor(p, 0,  0, 0, 0,  16, 42);    /* A on C0 addr 0 */
    place_tensor(p, 1,  1, 1, 0,  16, 99);    /* B on C1 addr 0 */

    p->output_op_id = 2;
    p->output_core  = 0;
    p->output_addr  = 0;
    p->output_size  = 16;
}

/* -------------------------------------------------------------- */
/* P1: matmul -> relu  chain                                       */
/*     A[16x16] @ C0,  B[16x16] @ C1                              */
/*     out = relu(A*B)  -> output @ C0:0                           */
/* -------------------------------------------------------------- */
static void load_p1(Problem *p)
{
    std::strcpy(p->name, "1_matmul_relu");
    p->num_cores = 4;  p->grid_rows = 2;  p->grid_cols = 2;

    p->num_ops = 4;
    set_input_op(p, 0, 256);                   /* A  16x16 */
    set_input_op(p, 1, 256);                   /* B  16x16 */
    set_matmul_op(p, 2, 0, 1, 16, 16, 16);    /* matmul   */
    set_relu_op(p, 3, 2, 256);                 /* relu     */

    p->num_initial_tensors = 2;
    place_tensor(p, 0,  0, 0, 0,  256, 7);
    place_tensor(p, 1,  1, 1, 0,  256, 13);

    p->output_op_id = 3;
    p->output_core  = 0;
    p->output_addr  = 0;
    p->output_size  = 256;
}

/* -------------------------------------------------------------- */
/* P2: two parallel matmuls -> concat                              */
/*     X[8x8] @ C0,  W1[8x8] @ C1,  W2[8x8] @ C3                */
/*     Y1 = X*W1,  Y2 = X*W2,  Z = concat(Y1, Y2)                */
/*     output Z (128 floats) @ C0:0                                */
/* -------------------------------------------------------------- */
static void load_p2(Problem *p)
{
    std::strcpy(p->name, "2_parallel_matmuls");
    p->num_cores = 4;  p->grid_rows = 2;  p->grid_cols = 2;

    p->num_ops = 6;
    set_input_op(p, 0, 64);                    /* X   8x8 */
    set_input_op(p, 1, 64);                    /* W1  8x8 */
    set_input_op(p, 2, 64);                    /* W2  8x8 */
    set_matmul_op(p, 3, 0, 1, 8, 8, 8);       /* Y1      */
    set_matmul_op(p, 4, 0, 2, 8, 8, 8);       /* Y2      */
    set_concat_op(p, 5, 3, 4, 128);           /* Z       */

    p->num_initial_tensors = 3;
    place_tensor(p, 0,  0, 0, 0,  64, 21);
    place_tensor(p, 1,  1, 1, 0,  64, 37);
    place_tensor(p, 2,  2, 3, 0,  64, 53);

    p->output_op_id = 5;
    p->output_core  = 0;
    p->output_addr  = 0;
    p->output_size  = 128;
}

/* -------------------------------------------------------------- */
/* P3: large matmul — benefits from splitting across cores         */
/*     A[32x32] @ C0,  B[32x32] @ C1                              */
/*     C = A*B   -> output @ C0:0                                  */
/* -------------------------------------------------------------- */
static void load_p3(Problem *p)
{
    std::strcpy(p->name, "3_large_matmul");
    p->num_cores = 4;  p->grid_rows = 2;  p->grid_cols = 2;

    p->num_ops = 3;
    set_input_op(p, 0, 1024);                  /* A 32x32 */
    set_input_op(p, 1, 1024);                  /* B 32x32 */
    set_matmul_op(p, 2, 0, 1, 32, 32, 32);    /* C = A*B */

    p->num_initial_tensors = 2;
    place_tensor(p, 0,  0, 0, 0,    1024, 100);
    place_tensor(p, 1,  1, 1, 0,    1024, 200);

    p->output_op_id = 2;
    p->output_core  = 0;
    p->output_addr  = 0;
    p->output_size  = 1024;
}

/* -------------------------------------------------------------- */
/* P4: mini neural network  (3x3 grid, 9 cores)                   */
/*                                                                 */
/*     X[8x16] @ C4       (center)                                 */
/*     W1[16x16] @ C0     matmul1 = X * W1                        */
/*     B1[8x16] @ C1      add1 = matmul1 + B1                     */
/*                         relu1 = relu(add1)                      */
/*     W2[16x8] @ C8      matmul2 = relu1 * W2                    */
/*                         relu2 = relu(matmul2)                   */
/*     output relu2 (64 floats) @ C0:0                             */
/* -------------------------------------------------------------- */
static void load_p4(Problem *p)
{
    std::strcpy(p->name, "4_mini_network");
    p->num_cores = 9;  p->grid_rows = 3;  p->grid_cols = 3;

    p->num_ops = 9;
    set_input_op(p, 0, 128);                       /* X   8x16  */
    set_input_op(p, 1, 256);                       /* W1  16x16 */
    set_input_op(p, 2, 128);                       /* B1  8x16  */
    set_input_op(p, 3, 128);                       /* W2  16x8  */
    set_matmul_op(p, 4, 0, 1, 8, 16, 16);         /* matmul1   */
    set_add_op(p, 5, 4, 2, 128);                   /* add1      */
    set_relu_op(p, 6, 5, 128);                     /* relu1     */
    set_matmul_op(p, 7, 6, 3, 8, 16, 8);          /* matmul2   */
    set_relu_op(p, 8, 7, 64);                      /* relu2     */

    p->num_initial_tensors = 4;
    place_tensor(p, 0,  0, 4, 0,   128, 300);  /* X  on C4 */
    place_tensor(p, 1,  1, 0, 0,   256, 400);  /* W1 on C0 */
    place_tensor(p, 2,  2, 1, 0,   128, 500);  /* B1 on C1 */
    place_tensor(p, 3,  3, 8, 0,   128, 600);  /* W2 on C8 */

    p->output_op_id = 8;
    p->output_core  = 0;
    p->output_addr  = 0;
    p->output_size  = 64;
}

/* -------------------------------------------------------------- */
/* P5: expensive moves: 4x4 mesh, DMA 3x more expensive           */
/*     X[8x16] @ C6:512    h = relu(X * W1 + B1)                   */
/*     W1[16x16] @ C0      y = h * W2                              */
/*     B1[8x16] @ C3:2048  out = y + X   (input added back)        */
/*     W2[16x16] @ C15                                             */
/*     output out (128 floats) @ C9:1024                           */
/* -------------------------------------------------------------- */
static void load_p5(Problem *p)
{
    std::strcpy(p->name, "5_expensive_moves");
    p->num_cores = 16;  p->grid_rows = 4;  p->grid_cols = 4;
    p->dma_factor = 3.0;

    p->num_ops = 9;
    set_input_op(p, 0, 128);                       /* X   8x16  */
    set_input_op(p, 1, 256);                       /* W1  16x16 */
    set_input_op(p, 2, 128);                       /* B1  8x16  */
    set_input_op(p, 3, 256);                       /* W2  16x16 */
    set_matmul_op(p, 4, 0, 1, 8, 16, 16);         /* X * W1    */
    set_add_op(p, 5, 4, 2, 128);                   /* + B1      */
    set_relu_op(p, 6, 5, 128);                     /* h         */
    set_matmul_op(p, 7, 6, 3, 8, 16, 16);         /* h * W2    */
    set_add_op(p, 8, 7, 0, 128);                   /* + X       */

    p->num_initial_tensors = 4;
    place_tensor(p, 0, 0,  6,  512, 128, 700);  /* X  on C6  */
    place_tensor(p, 1, 1,  0,    0, 256, 710);  /* W1 on C0  */
    place_tensor(p, 2, 2,  3, 2048, 128, 720);  /* B1 on C3  */
    place_tensor(p, 3, 3, 15,    0, 256, 730);  /* W2 on C15 */

    p->output_op_id = 8;
    p->output_core  = 9;
    p->output_addr  = 1024;
    p->output_size  = 128;
}

/* -------------------------------------------------------------- */
void load_problem(int test_id, Problem *p)
{
    std::memset(p, 0, sizeof(*p));
    p->dma_factor = 1.0;
    switch (test_id) {
    case 0: load_p0(p); break;
    case 1: load_p1(p); break;
    case 2: load_p2(p); break;
    case 3: load_p3(p); break;
    case 4: load_p4(p); break;
    case 5: load_p5(p); break;
    default:
        std::fprintf(stderr, "Unknown test_id %d\n", test_id);
        std::exit(1);
    }
    compute_expected_output(p);
}

/* ================================================================
 * Threaded Simulator
 * ================================================================
 *
 * Each core runs in its own std::thread, with malloc'd SRAM.
 * DMA copies data between cores using memcpy.
 * Signal/wait use a shared mutex + condition_variable.
 *
 * Synchronisation is enforced by real threading:
 *   - Without proper signal/wait, a DMA may read data that the
 *     source core hasn't computed yet, producing wrong results.
 *   - signal() publishes the core's current SRAM state (release).
 *   - wait() blocks until the signal fires, then proceeds (acquire).
 *
 * Cycle costs are tracked identically to the sequential model:
 *   matmul  = M * K * N
 *   relu    = size
 *   add     = size
 *   concat  = len_a + len_b
 *   dma     = dma_cost(): num_elements * distance * dma_factor
 *   signal  = 0   (records current cycle count)
 *   wait    = 0   (advances clock to signal-fire time if later)
 * ================================================================ */

/* Shared state between all core threads */
struct SharedState {
    std::mutex              mtx;
    std::condition_variable cv;
    bool                    sig_fired[MAX_SIGNALS];
    int                     sig_time[MAX_SIGNALS];
    std::atomic<bool>       abort_flag;

    SharedState() : abort_flag(false) {
        std::memset(sig_fired, 0, sizeof(sig_fired));
        std::memset(sig_time,  0, sizeof(sig_time));
    }
};

/* Forward declaration */
class Core;

class Core {
public:
    int     id;
    float  *sram;               /* malloc'd, SRAM_FLOATS elements           */
    int     cycles;             /* simulated cycle counter (for scoring)    */
    int     dma_cycles;         /* accumulated DMA cycle cost               */
    int     compute_cycles;     /* accumulated compute cycle cost           */

    /* set by simulate() before launching the thread */
    const Problem  *prob;
    int             num_cores;
    Core          **all_cores;  /* array of pointers to every core          */
    const Instruction *queue;
    int             num_instr;
    SharedState    *shared;

    bool    has_error;
    char    error_msg[256];

    Core(int core_id)
        : id(core_id), cycles(0), dma_cycles(0), compute_cycles(0),
          prob(nullptr), num_cores(0), all_cores(nullptr),
          queue(nullptr), num_instr(0), shared(nullptr),
          has_error(false)
    {
        sram = (float *)std::calloc(SRAM_FLOATS, sizeof(float));
        if (!sram) {
            std::fprintf(stderr, "out of memory allocating SRAM for core %d\n", id);
            std::exit(1);
        }
        error_msg[0] = '\0';
    }

    ~Core() { std::free(sram); }

    /* ---- main entry point, runs in its own thread ---- */
    void run()
    {
        for (int pc = 0; pc < num_instr; pc++) {
            if (has_error || shared->abort_flag.load(std::memory_order_relaxed))
                return;
            const Instruction &inst = queue[pc];
            switch (inst.opcode) {
            case OP_MATMUL: exec_matmul(inst); break;
            case OP_RELU:   exec_relu(inst);   break;
            case OP_ADD:    exec_add(inst);    break;
            case OP_CONCAT: exec_concat(inst); break;
            case OP_DMA:    exec_dma(inst);    break;
            case OP_SIGNAL: exec_signal(inst); break;
            case OP_WAIT:   exec_wait(inst);   break;
            }
        }
    }

    /* ---- compute routines (operate on local SRAM only) ---- */

    void exec_matmul(const Instruction &inst)
    {
        int M = inst.matmul.M, K = inst.matmul.K, N = inst.matmul.N;
        int ia = inst.matmul.in_addr;
        int wa = inst.matmul.weight_addr;
        int oa = inst.matmul.out_addr;
        if (ia + M*K > SRAM_FLOATS || wa + K*N > SRAM_FLOATS ||
            oa + M*N > SRAM_FLOATS) {
            set_error("core %d: matmul SRAM overflow", id);
            return;
        }
        ref_matmul(&sram[ia], &sram[wa], &sram[oa], M, K, N);
        int cost = M * K * N;
        cycles += cost;
        compute_cycles += cost;
    }

    void exec_relu(const Instruction &inst)
    {
        int sz = inst.relu.size;
        int ia = inst.relu.in_addr;
        int oa = inst.relu.out_addr;
        if (ia + sz > SRAM_FLOATS || oa + sz > SRAM_FLOATS) {
            set_error("core %d: relu SRAM overflow", id);
            return;
        }
        ref_relu(&sram[ia], &sram[oa], sz);
        cycles += sz;
        compute_cycles += sz;
    }

    void exec_add(const Instruction &inst)
    {
        int sz = inst.add.size;
        if (inst.add.a_addr + sz > SRAM_FLOATS ||
            inst.add.b_addr + sz > SRAM_FLOATS ||
            inst.add.out_addr + sz > SRAM_FLOATS) {
            set_error("core %d: add SRAM overflow", id);
            return;
        }
        ref_add(&sram[inst.add.a_addr], &sram[inst.add.b_addr],
                &sram[inst.add.out_addr], sz);
        cycles += sz;
        compute_cycles += sz;
    }

    void exec_concat(const Instruction &inst)
    {
        int la = inst.concat.len_a, lb = inst.concat.len_b;
        if (inst.concat.a_addr + la > SRAM_FLOATS ||
            inst.concat.b_addr + lb > SRAM_FLOATS ||
            inst.concat.out_addr + la + lb > SRAM_FLOATS) {
            set_error("core %d: concat SRAM overflow", id);
            return;
        }
        ref_concat(&sram[inst.concat.a_addr], &sram[inst.concat.b_addr],
                   &sram[inst.concat.out_addr], la, lb);
        int cost = la + lb;
        cycles += cost;
        compute_cycles += cost;
    }

    /* ---- DMA: memcpy from another core's SRAM ---- */

    void exec_dma(const Instruction &inst)
    {
        int sc = inst.dma.src_core;
        int sa = inst.dma.src_addr;
        int da = inst.dma.dst_addr;
        int ne = inst.dma.num_elements;
        if (sc < 0 || sc >= num_cores) {
            set_error("core %d: DMA from invalid core %d", id, sc);
            return;
        }
        if (sa + ne > SRAM_FLOATS || da + ne > SRAM_FLOATS) {
            set_error("core %d: DMA out of SRAM bounds", id);
            return;
        }
        /* memcpy from source core's SRAM into local SRAM */
        std::memcpy(&sram[da], &all_cores[sc]->sram[sa],
                    ne * sizeof(float));
        int cost = dma_cost(prob, id, sc, ne);
        cycles += cost;
        dma_cycles += cost;
    }

    /* ---- signal: publish SRAM state, wake waiters ---- */

    void exec_signal(const Instruction &inst)
    {
        int sid = inst.signal.id;
        {
            std::lock_guard<std::mutex> lock(shared->mtx);
            shared->sig_time[sid]  = cycles;
            shared->sig_fired[sid] = true;
        }
        shared->cv.notify_all();
        /* signal is zero-cost */
    }

    /* ---- wait: block until signal fires, advance clock ---- */

    void exec_wait(const Instruction &inst)
    {
        int sid = inst.wait.id;
        std::unique_lock<std::mutex> lock(shared->mtx);
        bool ok = shared->cv.wait_for(lock, std::chrono::seconds(5), [&]{
            return shared->sig_fired[sid] ||
                   shared->abort_flag.load(std::memory_order_relaxed);
        });
        if (shared->abort_flag.load(std::memory_order_relaxed))
            return;
        if (!ok) {
            lock.unlock();
            set_error("core %d: deadlock waiting for signal %d", id, sid);
            return;
        }
        if (shared->sig_time[sid] > cycles)
            cycles = shared->sig_time[sid];
        /* wait is zero-cost */
    }

private:
    void set_error(const char *fmt, ...)
    {
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(error_msg, sizeof(error_msg), fmt, ap);
        va_end(ap);
        has_error = true;
        shared->abort_flag.store(true, std::memory_order_release);
        shared->cv.notify_all();
    }
};

/* ================================================================
 * simulate()  — create Core objects, launch threads, collect results
 * ================================================================ */
int simulate(const Problem *prob, const Program *prog, SimResult *res)
{
    std::memset(res, 0, sizeof(*res));
    res->valid   = 0;
    res->correct = 0;
    int nc = prob->num_cores;

    /* create cores with malloc'd SRAM */
    std::vector<Core *> cores(nc);
    for (int c = 0; c < nc; c++)
        cores[c] = new Core(c);

    /* shared synchronisation state */
    SharedState shared;

    /* configure each core */
    Core **core_ptrs = cores.data();
    for (int c = 0; c < nc; c++) {
        cores[c]->prob       = prob;
        cores[c]->num_cores  = nc;
        cores[c]->all_cores  = core_ptrs;
        cores[c]->queue      = prog->queues[c];
        cores[c]->num_instr  = prog->num_instr[c];
        cores[c]->shared     = &shared;
    }

    /* load initial tensors into core SRAMs */
    for (int t = 0; t < prob->num_initial_tensors; t++) {
        const InitialTensor &it = prob->initial_tensors[t];
        if (it.sram_addr + it.num_floats > SRAM_FLOATS) {
            std::snprintf(res->error, sizeof(res->error),
                         "initial tensor op%d overflows SRAM on core %d",
                         it.op_id, it.core_id);
            for (int c = 0; c < nc; c++) delete cores[c];
            return 0;
        }
        std::memcpy(&cores[it.core_id]->sram[it.sram_addr],
                    it.data, it.num_floats * sizeof(float));
    }

    /* launch one thread per core */
    std::vector<std::thread> threads;
    for (int c = 0; c < nc; c++)
        threads.emplace_back(&Core::run, cores[c]);

    /* wait for all threads to finish */
    for (auto &t : threads)
        t.join();

    /* check for errors from any core */
    for (int c = 0; c < nc; c++) {
        if (cores[c]->has_error) {
            std::strncpy(res->error, cores[c]->error_msg,
                        sizeof(res->error) - 1);
            /* copy SRAM into result for debugging */
            for (int cc = 0; cc < nc; cc++)
                std::memcpy(res->sram[cc], cores[cc]->sram,
                           SRAM_FLOATS * sizeof(float));
            for (int cc = 0; cc < nc; cc++) delete cores[cc];
            return 0;
        }
    }

    /* collect timing results */
    res->valid = 1;
    res->makespan = 0;
    res->total_dma_cycles = 0;
    res->total_compute_cycles = 0;
    for (int c = 0; c < nc; c++) {
        res->core_cycles[c] = cores[c]->cycles;
        if (cores[c]->cycles > res->makespan)
            res->makespan = cores[c]->cycles;
        res->total_dma_cycles     += cores[c]->dma_cycles;
        res->total_compute_cycles += cores[c]->compute_cycles;
        /* copy SRAM into result for verification */
        std::memcpy(res->sram[c], cores[c]->sram,
                   SRAM_FLOATS * sizeof(float));
    }

    for (int c = 0; c < nc; c++) delete cores[c];
    return 1;
}

/* ================================================================
 * Output Verification
 * ================================================================ */
int verify_output(const Problem *prob, const SimResult *res)
{
    const float *actual = &res->sram[prob->output_core][prob->output_addr];
    const float *expect = prob->expected_output;
    for (int i = 0; i < prob->output_size; i++) {
        float diff = std::fabs(actual[i] - expect[i]);
        float mag  = std::fabs(expect[i]) + 1e-6f;
        if (diff / mag > 1e-3f) {
            return 0;
        }
    }
    return 1;
}

/* ================================================================
 * Pretty-printing
 * ================================================================ */
static const char *gtype_str(GraphOpType t)
{
    switch (t) {
    case GRAPH_INPUT:  return "INPUT ";
    case GRAPH_MATMUL: return "MATMUL";
    case GRAPH_RELU:   return "RELU  ";
    case GRAPH_ADD:    return "ADD   ";
    case GRAPH_CONCAT: return "CONCAT";
    }
    return "???   ";
}

void print_problem(const Problem *p)
{
    std::printf("=== Problem: %s ===\n", p->name);
    std::printf("Hardware : %d cores  (%d x %d mesh)\n",
           p->num_cores, p->grid_rows, p->grid_cols);
    std::printf("DMA cost : floats x distance x %g\n", p->dma_factor);
    std::printf("SRAM/core: %d floats (%d KB)\n",
           SRAM_FLOATS, (int)(SRAM_FLOATS * sizeof(float) / 1024));
    std::printf("\n");

    std::printf("Core layout:\n");
    for (int r = 0; r < p->grid_rows; r++) {
        std::printf("  ");
        for (int c = 0; c < p->grid_cols; c++) {
            if (c) std::printf(" -- ");
            std::printf("C%d", r * p->grid_cols + c);
        }
        std::printf("\n");
        if (r < p->grid_rows - 1) {
            std::printf("  ");
            for (int c = 0; c < p->grid_cols; c++) {
                if (c) std::printf("    ");
                std::printf("|  ");
            }
            std::printf("\n");
        }
    }
    std::printf("\n");

    std::printf("Compute graph (%d ops):\n", p->num_ops);
    for (int i = 0; i < p->num_ops; i++) {
        const GraphOp &op = p->ops[i];
        std::printf("  Op%-3d  %s  out_size=%-5d", op.id, gtype_str(op.type),
               op.output_size);
        if (op.type == GRAPH_MATMUL)
            std::printf("  [%d x %d] * [%d x %d]", op.M, op.K, op.K, op.N);
        if (op.num_inputs > 0) {
            std::printf("  <- [");
            for (int j = 0; j < op.num_inputs; j++) {
                if (j) std::printf(", ");
                std::printf("Op%d", op.input_ids[j]);
            }
            std::printf("]");
        }
        std::printf("\n");
    }
    std::printf("\n");

    std::printf("Initial data placement:\n");
    for (int t = 0; t < p->num_initial_tensors; t++) {
        const InitialTensor &it = p->initial_tensors[t];
        std::printf("  Op%-3d  -> Core %d  addr [%d..%d)  (%d floats)\n",
               it.op_id, it.core_id,
               it.sram_addr, it.sram_addr + it.num_floats,
               it.num_floats);
    }
    std::printf("\n");
    std::printf("Output: Op%d  must be at Core %d addr [%d..%d)  (%d floats)\n",
           p->output_op_id, p->output_core,
           p->output_addr, p->output_addr + p->output_size,
           p->output_size);
    std::printf("\n");
}

void print_result(const SimResult *res, int num_cores)
{
    if (!res->valid) {
        std::printf("  INVALID: %s\n", res->error);
        return;
    }
    std::printf("  Makespan      : %d cycles\n", res->makespan);
    std::printf("  Compute cycles: %d\n", res->total_compute_cycles);
    std::printf("  DMA cycles    : %d\n", res->total_dma_cycles);
    std::printf("  Per-core:\n");
    for (int c = 0; c < num_cores; c++) {
        if (res->core_cycles[c] == 0) continue;
        int util = (res->makespan > 0)
                   ? (int)(100.0 * res->core_cycles[c] / res->makespan)
                   : 0;
        std::printf("    Core %2d: %6d cycles  (%3d%%)\n",
               c, res->core_cycles[c], util);
    }
    if (res->sync_warnings > 0)
        std::printf("  Sync warnings: %d (unsynchronised DMA reads)\n",
                    res->sync_warnings);
    if (res->correct)
        std::printf("  Output: CORRECT\n");
    else
        std::printf("  Output: *** INCORRECT ***\n");
}

/* ================================================================
 * Main
 * ================================================================ */
int main(int argc, char **argv)
{
    int run_all = 0;
    int test_id = 0;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "all") == 0)       run_all = 1;
        else if (std::strcmp(argv[i], "-v") == 0)   verbose = 1;
        else test_id = std::atoi(argv[i]);
    }

    int first = run_all ? 0 : test_id;
    int last  = run_all ? NUM_PROBLEMS - 1 : test_id;

    int total_score = 0;
    int total_possible = 0;

    for (int t = first; t <= last; t++) {
        Problem prob;
        load_problem(t, &prob);

        if (verbose) print_problem(&prob);

        Program prog;
        std::memset(&prog, 0, sizeof(prog));

        solve(&prob, &prog);

        SimResult res;
        simulate(&prob, &prog, &res);

        if (res.valid)
            res.correct = verify_output(&prob, &res);

        std::printf("[%s]\n", prob.name);
        print_result(&res, prob.num_cores);

        if (res.valid && res.correct) total_score++;
        total_possible++;

        /* free initial tensor data & expected output */
        for (int i = 0; i < prob.num_initial_tensors; i++)
            std::free(prob.initial_tensors[i].data);
        std::free(prob.expected_output);

        std::printf("\n");
    }

    if (run_all) {
        std::printf("========================================\n");
        std::printf("Correct: %d / %d\n", total_score, total_possible);
        std::printf("========================================\n");
    }

    return (total_score == total_possible) ? 0 : 1;
}
