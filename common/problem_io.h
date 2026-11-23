/*
 * problem_io.h — Problems described in a text file, and random problems.
 *
 * Lets the judge server and the check tool work on problems other
 * than the six built into hackathon.cpp.
 *
 * File format (one keyword per line, '#' starts a comment):
 *
 *   problem <id> <name>
 *   grid <rows> <cols>                       (square: rows == cols)
 *   dma_factor <x>                           (optional, 0.1 to 16; default 1)
 *   input  <op> <size> <core> <addr> <seed>
 *   matmul <op> <a> <b> <M> <K> <N>
 *   relu   <op> <a>
 *   add    <op> <a> <b>
 *   concat <op> <a> <b>
 *   output <op> <core> <addr>
 *   end
 *
 * Ops are numbered 0, 1, 2... in order and may only use earlier ops.
 * Input data is generated from <seed> like hackathon.cpp does.
 */
#ifndef PROBLEM_IO_H
#define PROBLEM_IO_H

#include "hackathon.h"
#include <memory>
#include <string>
#include <vector>

/** Problem owning its tensor data. */
struct OwnedProblem {
    int     id = 0;
    Problem p;

    OwnedProblem() { std::memset(&p, 0, sizeof(p)); }
    ~OwnedProblem();
    OwnedProblem(const OwnedProblem &) = delete;
    OwnedProblem &operator=(const OwnedProblem &) = delete;
};

using ProblemList = std::vector<std::unique_ptr<OwnedProblem>>;

/**
 * Reads every problem of a file and computes its expected output.
 *
 * @return empty string on success, otherwise "line N: reason"
 */
std::string read_problems(FILE *f, ProblemList *out);

/** Reads a problem file by path; exits with a message on error. */
ProblemList load_problem_file(const char *path);

/** Fills p->expected_output with the reference result. */
void compute_expected(Problem *p);

/**
 * Builds a random problem: square grid (2x2, 3x3 or 4x4), a chain of matmul /
 * add / relu layers with parallel branches, inputs and output on any
 * core, DMA factor 1 to 3. Everything fits in one core's SRAM, so baseline.cpp solves it.
 */
void random_problem(unsigned seed, OwnedProblem *out);

#endif /* PROBLEM_IO_H */
