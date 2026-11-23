/*
 * score.cpp — Checks and scores one program, as the judge does.
 */
#include "score.h"
#include <set>

static bool span_ok(long long addr, long long len)
{
    return addr >= 0 && len >= 0 && addr + len <= SRAM_FLOATS;
}

std::string validate_program(const Problem *prob, const Program *p)
{
    char err[160];
    std::set<int> fired;
    for (int c = 0; c < MAX_CORES; c++) {
        if (p->num_instr[c] < 0 || p->num_instr[c] > MAX_INSTR_PER_CORE) {
            std::snprintf(err, sizeof(err), "core %d: bad instruction count", c);
            return err;
        }
        if (p->num_instr[c] && c >= prob->num_cores) {
            std::snprintf(err, sizeof(err), "core %d does not exist", c);
            return err;
        }
        for (int i = 0; i < p->num_instr[c]; i++) {
            const Instruction &in = p->queues[c][i];
            bool ok = true;
            switch (in.opcode) {
            case OP_MATMUL: {
                long long M = in.matmul.M, K = in.matmul.K, N = in.matmul.N;
                ok = M > 0 && K > 0 && N > 0 &&
                     span_ok(in.matmul.in_addr, M * K) &&
                     span_ok(in.matmul.weight_addr, K * N) &&
                     span_ok(in.matmul.out_addr, M * N);
                break;
            }
            case OP_RELU:
                ok = span_ok(in.relu.in_addr, in.relu.size) &&
                     span_ok(in.relu.out_addr, in.relu.size);
                break;
            case OP_ADD:
                ok = span_ok(in.add.a_addr, in.add.size) &&
                     span_ok(in.add.b_addr, in.add.size) &&
                     span_ok(in.add.out_addr, in.add.size);
                break;
            case OP_CONCAT:
                ok = span_ok(in.concat.a_addr, in.concat.len_a) &&
                     span_ok(in.concat.b_addr, in.concat.len_b) &&
                     span_ok(in.concat.out_addr,
                             (long long)in.concat.len_a + in.concat.len_b);
                break;
            case OP_DMA:
                ok = in.dma.src_core >= 0 &&
                     in.dma.src_core < prob->num_cores &&
                     span_ok(in.dma.src_addr, in.dma.num_elements) &&
                     span_ok(in.dma.dst_addr, in.dma.num_elements);
                break;
            case OP_SIGNAL:
                ok = in.signal.id >= 0 && in.signal.id < MAX_SIGNALS;
                if (ok && !fired.insert(in.signal.id).second) {
                    std::snprintf(err, sizeof(err),
                                  "signal %d fired more than once",
                                  in.signal.id);
                    return err;
                }
                break;
            case OP_WAIT:
                ok = in.wait.id >= 0 && in.wait.id < MAX_SIGNALS;
                break;
            default:
                ok = false;
            }
            if (!ok) {
                std::snprintf(err, sizeof(err),
                              "core %d instr %d: argument out of range", c, i);
                return err;
            }
        }
    }
    return "";
}

Score score_program(const Problem *prob, const Program *p, int runs)
{
    Score s;
    s.error = validate_program(prob, p);
    if (s.error.empty()) {
        s.r = replay(prob, p);
        if (s.r.deadlock) s.error = "deadlock: a wait never gets its signal";
    }
    if (!s.error.empty()) {
        s.status = "invalid";
        return s;
    }
    s.sim.reset(new SimResult());
    SimResult *sim = s.sim.get();
    bool correct = true;
    for (int run = 0; run < runs; run++) {
        simulate(prob, p, sim);
        if (!sim->valid) { s.error = sim->error; correct = false; break; }
        sim->correct = verify_output(prob, sim);
        if (!sim->correct) { correct = false; break; }
    }
    if (sim->valid && sim->makespan != s.r.makespan) {
        s.error = "internal: replay and simulate disagree";
        correct = false;
    }
    s.status = correct ? "correct" : (sim->valid ? "incorrect" : "invalid");
    if (s.status == "incorrect")
        s.error = "wrong output (missing signal/wait before a DMA?)";
    s.makespan = s.r.makespan;
    if (sim->valid && s.makespan > 0)
        s.util = 100.0 * sim->total_compute_cycles /
                 ((double)prob->num_cores * s.makespan);
    return s;
}

int baseline_makespan(const Problem *prob)
{
    std::unique_ptr<Program> p(new Program());
    std::memset(p.get(), 0, sizeof(Program));
    baseline_solve(prob, p.get());
    if (!validate_program(prob, p.get()).empty()) return 0;
    Replay r = replay(prob, p.get());
    return r.deadlock ? 0 : r.makespan;
}
