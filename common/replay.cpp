/*
 * replay.cpp — Deterministic replay of the cost model and JSON trace output.
 */
#include "replay.h"
#include <cstdarg>

static std::string fmt(const char *f, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

Replay replay(const Problem *prob, const Program *prog)
{
    Replay r;
    int nc = prob->num_cores;
    std::vector<int> t(nc, 0), pc(nc, 0);
    std::vector<int> sig_time(MAX_SIGNALS, -1);

    bool progress = true;
    while (progress) {
        progress = false;
        for (int c = 0; c < nc; c++) {
            while (pc[c] < prog->num_instr[c]) {
                const Instruction &in = prog->queues[c][pc[c]];
                Event e;
                e.core = c;
                e.start = t[c];
                int cost = 0;
                switch (in.opcode) {
                case OP_MATMUL:
                    cost = in.matmul.M * in.matmul.K * in.matmul.N;
                    e.kind = "compute";
                    e.label = fmt("matmul %dx%dx%d", in.matmul.M,
                                  in.matmul.K, in.matmul.N);
                    break;
                case OP_RELU:
                    cost = in.relu.size;
                    e.kind = "compute";
                    e.label = fmt("relu %d", in.relu.size);
                    break;
                case OP_ADD:
                    cost = in.add.size;
                    e.kind = "compute";
                    e.label = fmt("add %d", in.add.size);
                    break;
                case OP_CONCAT:
                    cost = in.concat.len_a + in.concat.len_b;
                    e.kind = "compute";
                    e.label = fmt("concat %d+%d", in.concat.len_a,
                                  in.concat.len_b);
                    break;
                case OP_DMA: {
                    cost = dma_cost(prob, c, in.dma.src_core,
                                    in.dma.num_elements);
                    e.kind = "dma";
                    e.src = in.dma.src_core;
                    e.src_addr = in.dma.src_addr;
                    e.dst_addr = in.dma.dst_addr;
                    e.n = in.dma.num_elements;
                    e.label = fmt("dma C%d->C%d %d floats", e.src, c, e.n);
                    break;
                }
                case OP_SIGNAL:
                    if (sig_time[in.signal.id] >= 0) r.reused_signals++;
                    sig_time[in.signal.id] = t[c];
                    e.kind = "signal";
                    e.label = fmt("signal %d", in.signal.id);
                    break;
                case OP_WAIT:
                    if (sig_time[in.wait.id] < 0)
                        goto blocked;
                    if (sig_time[in.wait.id] > t[c])
                        cost = sig_time[in.wait.id] - t[c];
                    e.kind = "wait";
                    e.label = fmt("wait %d", in.wait.id);
                    break;
                }
                t[c] += cost;
                e.end = t[c];
                if (e.kind == "signal" || e.end > e.start)
                    r.events.push_back(e);
                pc[c]++;
                progress = true;
            }
        blocked:;
        }
    }
    for (int c = 0; c < nc; c++) {
        if (pc[c] < prog->num_instr[c]) r.deadlock = true;
        r.core_end[c] = t[c];
        if (t[c] > r.makespan) r.makespan = t[c];
    }
    return r;
}

static const char *gtype(GraphOpType t)
{
    switch (t) {
    case GRAPH_INPUT:  return "input";
    case GRAPH_MATMUL: return "matmul";
    case GRAPH_RELU:   return "relu";
    case GRAPH_ADD:    return "add";
    case GRAPH_CONCAT: return "concat";
    }
    return "?";
}

void write_problem_json(FILE *f, const Problem *prob, const Replay &r,
                        const SimResult *sim, const char *name_suffix)
{
    std::fprintf(f, "{\"name\":\"%s%s\",\"rows\":%d,\"cols\":%d,"
                    "\"dma_factor\":%g,",
                 prob->name, name_suffix,
                 prob->grid_rows, prob->grid_cols,
                 prob->dma_factor > 0 ? prob->dma_factor : 1.0);
    std::fprintf(f, "\"makespan\":%d,\"sim_makespan\":%d,\"valid\":%s,"
                    "\"correct\":%s,\"deadlock\":%s,\"error\":\"%s\",",
                 r.makespan, sim->makespan, sim->valid ? "true" : "false",
                 sim->correct ? "true" : "false",
                 r.deadlock ? "true" : "false", sim->error);
    std::fprintf(f, "\"output\":{\"core\":%d,\"addr\":%d,\"size\":%d},",
                 prob->output_core, prob->output_addr, prob->output_size);

    std::fprintf(f, "\"graph\":[");
    for (int i = 0; i < prob->num_ops; i++) {
        const GraphOp &o = prob->ops[i];
        std::fprintf(f, "%s{\"id\":%d,\"type\":\"%s\",\"size\":%d,\"inputs\":[",
                     i ? "," : "", o.id, gtype(o.type), o.output_size);
        for (int j = 0; j < o.num_inputs; j++)
            std::fprintf(f, "%s%d", j ? "," : "", o.input_ids[j]);
        std::fprintf(f, "]}");
    }
    std::fprintf(f, "],\"tensors\":[");
    for (int i = 0; i < prob->num_initial_tensors; i++) {
        const InitialTensor &it = prob->initial_tensors[i];
        std::fprintf(f, "%s{\"op\":%d,\"core\":%d,\"addr\":%d,\"n\":%d}",
                     i ? "," : "", it.op_id, it.core_id, it.sram_addr,
                     it.num_floats);
    }
    std::fprintf(f, "],\"events\":[");
    for (size_t i = 0; i < r.events.size(); i++) {
        const Event &e = r.events[i];
        std::fprintf(f, "%s{\"core\":%d,\"start\":%d,\"end\":%d,"
                        "\"kind\":\"%s\",\"label\":\"%s\"",
                     i ? "," : "", e.core, e.start, e.end,
                     e.kind.c_str(), e.label.c_str());
        if (e.kind == "dma")
            std::fprintf(f, ",\"src\":%d,\"src_addr\":%d,\"dst_addr\":%d,"
                            "\"n\":%d", e.src, e.src_addr, e.dst_addr, e.n);
        std::fprintf(f, "}");
    }
    std::fprintf(f, "]}");
}
