/*
 * score.h — Checks and scores one program, as the judge does.
 *
 * Shared by the judge server and tools/check.cpp.
 */
#ifndef SCORE_H
#define SCORE_H

#include "replay.h"
#include <memory>
#include <string>

/** Outcome of one problem. */
struct Score {
    std::string status;     /**< correct | incorrect | invalid | missing */
    std::string error;
    int makespan = 0;
    int baseline = 0;
    double util = 0;        /**< compute cycles / (cores x makespan), in % */
    Replay r;                         /**< valid unless status is invalid */
    std::unique_ptr<SimResult> sim;   /**< last simulation, if any */
};

/**
 * Returns an empty string if the program is safe to simulate
 * (hackathon.cpp trusts its input), otherwise the reason.
 */
std::string validate_program(const Problem *prob, const Program *p);

/**
 * Validates, checks for deadlocks with the replay, then simulates
 * @p runs times; every run must produce the expected output.
 */
Score score_program(const Problem *prob, const Program *p, int runs = 3);

/** baseline.cpp's solve(), built with -Dsolve=baseline_solve. */
void baseline_solve(const Problem *prob, Program *prog);

/** Makespan of baseline_solve() on @p prob (0 if it is not valid). */
int baseline_makespan(const Problem *prob);

#endif /* SCORE_H */
