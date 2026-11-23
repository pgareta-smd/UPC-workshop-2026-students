/*
 * trace.cpp — Dumps a per-core timeline of a schedule for viz/index.html.
 *
 * Links against hackathon.cpp (built with -Dmain=hackathon_main) and a
 * solve() implementation (solution.cpp or baseline.cpp). For every
 * problem it replays the cost model instruction by instruction, records
 * compute / dma / wait intervals, cross-checks the makespan with
 * simulate() and writes everything to viz/trace.js.
 *
 * Usage: build/trace [all|<id>] [--demo] [-o <file>]
 *   --demo  use a built-in parallel schedule for problem 2
 */
#include "replay.h"
#include <memory>

#ifndef TRACE_OUT
#define TRACE_OUT "trace.js"
#endif

/* ---------------------------------------------------------------- */

/** P2 split over C1 and C3, gathered on C0. */
static void demo_p2(Program *p)
{
    emit_dma(p, 1, 0, 0, 64, 64);
    emit_matmul(p, 1, 64, 0, 128, 8, 8, 8);
    emit_signal(p, 1, 1);
    emit_dma(p, 3, 0, 0, 64, 64);
    emit_matmul(p, 3, 64, 0, 128, 8, 8, 8);
    emit_signal(p, 3, 2);
    emit_wait(p, 0, 1);
    emit_dma(p, 0, 1, 128, 256, 64);
    emit_wait(p, 0, 2);
    emit_dma(p, 0, 3, 128, 320, 64);
    emit_concat(p, 0, 256, 320, 0, 64, 64);
}

int main(int argc, char **argv)
{
    int first = 0, last = NUM_PROBLEMS - 1;
    bool demo = false;
    const char *out = TRACE_OUT;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--demo")) demo = true;
        else if (!std::strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (std::strcmp(argv[i], "all")) first = last = std::atoi(argv[i]);
    }

    FILE *f = std::fopen(out, "w");
    if (!f) { std::perror(out); return 1; }
    std::fprintf(f, "window.TRACE = {\"problems\":[");

    int mismatches = 0;
    for (int t = first; t <= last; t++) {
        Problem prob;
        load_problem(t, &prob);
        std::unique_ptr<Program> prog(new Program());
        std::memset(prog.get(), 0, sizeof(Program));
        bool use_demo = demo && t == 2;
        if (use_demo) demo_p2(prog.get());
        else solve(&prob, prog.get());

        Replay r = replay(&prob, prog.get());
        std::unique_ptr<SimResult> sim(new SimResult());
        simulate(&prob, prog.get(), sim.get());
        if (sim->valid) sim->correct = verify_output(&prob, sim.get());

        bool match = !sim->valid || r.makespan == sim->makespan;
        if (!match) mismatches++;
        std::fprintf(stderr, "%-22s makespan %6d  sim %6d  %s%s%s\n",
                     prob.name, r.makespan, sim->makespan,
                     match ? "ok" : "MISMATCH",
                     sim->valid ? (sim->correct ? "  correct" : "  INCORRECT")
                                : "  INVALID",
                     r.reused_signals ? "  (reused signal ids)" : "");

        if (t != first) std::fprintf(f, ",");
        write_problem_json(f, &prob, r, sim.get(), use_demo ? " (demo)" : "");

        for (int i = 0; i < prob.num_initial_tensors; i++)
            std::free(prob.initial_tensors[i].data);
        std::free(prob.expected_output);
    }
    std::fprintf(f, "]};\n");
    std::fclose(f);
    std::fprintf(stderr, "wrote %s\n", out);
    return mismatches ? 1 : 0;
}
