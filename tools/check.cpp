/*
 * check.cpp — Runs your solve() like the server does.
 *
 * Scores solution.cpp on the problems of a problem file (the public ones
 * by default) or on random problems, to check that it works on problems
 * it has never seen: the final round uses hidden problems.
 *
 * Usage: build/check [<problem file>] [--random <n>] [--seed <s>] [--trace]
 *   --random <n>  n random problems instead of a file
 *   --seed <s>    first random seed (default 1)
 *   --trace       also write the timelines to viz/trace.js
 * Exit status: 0 if every problem is correct.
 */
#include "problem_io.h"
#include "score.h"

#ifndef PUBLIC_PROBLEMS
#define PUBLIC_PROBLEMS "problems/public.txt"
#endif
#ifndef TRACE_OUT
#define TRACE_OUT "trace.js"
#endif

int main(int argc, char **argv)
{
    const char *file = PUBLIC_PROBLEMS;
    int random = 0;
    unsigned seed = 1;
    bool trace = false;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--random") && i + 1 < argc)
            random = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = (unsigned)std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--trace"))
            trace = true;
        else if (argv[i][0] != '-')
            file = argv[i];
        else {
            std::fprintf(stderr, "usage: %s [<problem file>] [--random <n>] "
                         "[--seed <s>] [--trace]\n", argv[0]);
            return 2;
        }
    }

    ProblemList probs;
    if (random > 0) {
        for (int i = 0; i < random; i++) {
            probs.emplace_back(new OwnedProblem());
            random_problem(seed + i, probs.back().get());
        }
    } else {
        probs = load_problem_file(file);
    }

    FILE *tf = nullptr;
    if (trace) {
        tf = std::fopen(TRACE_OUT, "w");
        if (!tf) { std::perror(TRACE_OUT); return 2; }
        std::fprintf(tf, "window.TRACE = {\"problems\":[");
    }

    std::printf("%-22s %-5s %-4s %-10s %10s %10s %8s %6s\n", "problem", "grid",
                "dma", "status", "makespan", "baseline", "speedup", "util");
    int correct = 0, traced = 0;
    long long total = 0;
    std::unique_ptr<Program> prog(new Program());
    for (auto &op : probs) {
        const Problem *p = &op->p;
        std::memset(prog.get(), 0, sizeof(Program));
        solve(p, prog.get());
        Score s = score_program(p, prog.get());
        int base = baseline_makespan(p);
        char grid[8];
        std::snprintf(grid, sizeof(grid), "%dx%d", p->grid_rows, p->grid_cols);
        char factor[24];
        std::snprintf(factor, sizeof(factor), "x%g", p->dma_factor);
        std::printf("%-22s %-5s %-4s %-10s", p->name, grid, factor, s.status.c_str());
        if (s.status == "correct") {
            correct++;
            total += s.makespan;
            std::printf(" %10d %10d %7.2fx %5.1f%%\n", s.makespan, base,
                        s.makespan ? (double)base / s.makespan : 0.0, s.util);
        } else {
            std::printf(" %10s %10d   %s\n", "", base, s.error.c_str());
        }
        if (tf && s.sim) {
            if (traced++) std::fputc(',', tf);
            write_problem_json(tf, p, s.r, s.sim.get(), "");
        }
    }
    std::printf("\n%d / %zu correct, total makespan %lld\n", correct,
                probs.size(), total);
    if (tf) {
        std::fprintf(tf, "]};\n");
        std::fclose(tf);
        std::printf("wrote %s (open viz/index.html)\n", TRACE_OUT);
    }
    return correct == (int)probs.size() ? 0 : 1;
}
