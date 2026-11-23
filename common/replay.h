/*
 * replay.h — Deterministic replay of the cost model and JSON trace output.
 *
 * Shared by viz/trace.cpp and the judge server.
 */
#ifndef REPLAY_H
#define REPLAY_H

#include "hackathon.h"
#include <string>
#include <vector>

/** One interval on a core's timeline. */
struct Event {
    int         core;
    int         start, end;
    std::string kind;      /**< compute | dma | wait | signal */
    std::string label;
    int         src = -1;  /**< dma source core */
    int         src_addr = 0, dst_addr = 0, n = 0;
};

/** Result of replaying a program with the documented cycle costs. */
struct Replay {
    std::vector<Event> events;
    int  makespan = 0;
    int  core_end[MAX_CORES] = {0};
    bool deadlock = false;     /**< some core never got its signal */
    int  reused_signals = 0;   /**< signal ids fired more than once */
};

/**
 * Replays @p prog on @p prob one instruction at a time.
 *
 * Deterministic as long as every signal id is fired once. The program
 * must already be validated (ids and core numbers in range).
 */
Replay replay(const Problem *prob, const Program *prog);

/**
 * Writes one problem entry of the viewer trace as a JSON object.
 *
 * @param name_suffix appended to the problem name (e.g. " (demo)")
 */
void write_problem_json(FILE *f, const Problem *prob, const Replay &r,
                        const SimResult *sim, const char *name_suffix);

#endif /* REPLAY_H */
