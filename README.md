# Operator Graph Scheduler — Hackathon Challenge

## Overview

You are writing a scheduler for a multi-core AI accelerator chip. The chip has
processing cores arranged in a **2D mesh**, each with its own **local SRAM**.
Your job: given a neural-network compute graph, emit low-level instructions to
each core so that the graph executes correctly and **as fast as possible**.

The key insight: **data locality matters**. Moving data between cores via DMA
costs cycles proportional to the tensor size and the Manhattan distance between
cores. Place operations near their data to avoid expensive transfers — or split
large operators across cores to exploit parallelism. Finding the right balance
is the challenge.

## Requirements

- A C++17 compiler (g++ ≥ 8 or clang ≥ 7), CMake ≥ 3.16 and a web browser.
  Nothing else: no libraries, no Python.

| System | Install |
|---|---|
| Ubuntu / Debian | `sudo apt install build-essential cmake` |
| macOS | `xcode-select --install` and `brew install cmake` |
| Windows | Use WSL (Ubuntu) and the Ubuntu line above |

## Hardware Model

```
2×2 mesh:             3×3 mesh:
  C0 -- C1              C0 -- C1 -- C2
  |     |               |     |     |
  C2 -- C3              C3 -- C4 -- C5
                          |     |     |
                          C6 -- C7 -- C8
```

Grids are always square: 2×2, 3×3 or 4×4. Cores are numbered row by row,
so on a 4×4 mesh C0–C3 is the top row and C15 the bottom-right corner.

- Each core has **16 KB of local SRAM** (4096 floats).
- Cores can only compute on **local** SRAM. To use remote data, you must
  **DMA** it to local memory first.
- Distance between cores = **Manhattan distance** on the grid.
- Each problem has a **DMA factor** (`prob->dma_factor`, usually 1): moving
  data between cores costs that many times more. A high factor rewards
  keeping data where it is.

## Instruction Set

Your `solve()` function emits instructions to per-core queues using these
helpers:

| Instruction | Cost (cycles) | Description |
|---|---|---|
| `emit_matmul(prog, core, in, weight, out, M, K, N)` | M × K × N | C\[M×N\] = A\[M×K\] · B\[K×N\], all local addrs |
| `emit_relu(prog, core, in, out, size)` | size | out\[i\] = max(0, in\[i\]) |
| `emit_add(prog, core, a, b, out, size)` | size | out\[i\] = a\[i\] + b\[i\] |
| `emit_concat(prog, core, a, b, out, len_a, len_b)` | len\_a + len\_b | out = a ∥ b |
| `emit_dma(prog, core, src_core, src_addr, dst_addr, n)` | n × distance × dma\_factor | Copy n floats from remote core's SRAM to local |
| `emit_signal(prog, core, signal_id)` | 0 | Fire a named event |
| `emit_wait(prog, core, signal_id)` | 0 | Block until event is fired |

**DMA notes:**
- DMA blocks the executing (destination) core for the duration. The source core
  is unaffected.
- DMA reads the source SRAM **as-is** at the time the instruction executes. Use
  `signal`/`wait` to ensure the producer has finished writing before you DMA.
- The cost is rounded to the nearest cycle; `dma_cost(prob, core, src_core, n)`
  gives it for you.
- Local copy (same core) costs 1 cycle per element, whatever the DMA factor.

## Synchronization

Cores run **simultaneously**, each processing its own instruction queue
sequentially. Use `signal`/`wait` pairs to coordinate:

```cpp
// Core 0 computes, then signals
emit_matmul(prog, 0, ...);
emit_signal(prog, 0, /*signal_id=*/1);

// Core 1 waits for the signal, then DMAs the result
emit_wait(prog, 1, /*signal_id=*/1);
emit_dma(prog, 1, /*src_core=*/0, ...);
```

A `wait` blocks the core until the corresponding `signal` has fired. If the
signal has already fired by the time `wait` executes, the core advances its
clock to max(current\_time, signal\_fire\_time) and continues immediately.

## Splitting Matmuls

You can decompose a large matmul into smaller ones across cores.

**Row split** (split the M dimension — easiest, data is contiguous):
```
A[M×K] · B[K×N]  →  A_top[M/2 × K] · B  and  A_bot[M/2 × K] · B
```
Both halves share the same B matrix. The output halves concatenate row-wise.

**Column split** (split the N dimension):
```
A[M×K] · B[K×N]  →  A · B_left[K × N/2]  and  A · B_right[K × N/2]
```
Requires extracting non-contiguous columns from B (harder addressing).

**K split** (split the reduction dimension):
```
A[M×K] · B[K×N]  →  A_left[M × K/2] · B_top[K/2 × N]
                   + A_right[M × K/2] · B_bot[K/2 × N]
```
Requires an `add` to combine partial results.

## Test Problems

| # | Name | Grid | Graph | Key Challenge |
|---|------|------|-------|---------------|
| 0 | `single_matmul` | 2×2 | A·B | Warmup: just DMA + compute |
| 1 | `matmul_relu` | 2×2 | relu(A·B) | Chain with row-split opportunity |
| 2 | `parallel_matmuls` | 2×2 | concat(X·W1, X·W2) | Two independent matmuls → parallelise |
| 3 | `large_matmul` | 2×2 | A\[32×32\]·B\[32×32\] | Must split to get speedup |
| 4 | `mini_network` | 3×3 | matmul → add → relu → matmul → relu | Locality + parallelism + multi-op |
| 5 | `expensive_moves` | 4×4 | relu(X·W1 + B1)·W2 + X | Two matmuls in a row, then add the input back; moving data costs 3× |

Run `./build/baseline -v <id>` to see the full problem description for any test case.

## Getting Started

```bash
# build (once); after editing solution.cpp just run: make -C build
mkdir -p build && cd build && cmake .. && make && cd ..

# run baseline on all problems
./build/baseline all

# run your solution
./build/solution all

# verbose output (shows graph + core layout)
./build/solution -v 3

# score your solve() like the server does
./build/check                    # the six public problems
./build/check --random 50        # 50 random problems (other grids and graphs)
./build/check --random 1 --seed 7 --trace   # one of them, timeline in viz/trace.js
```

## Seeing your schedule

```bash
./build/check --trace          # score solution.cpp and save its timelines
./build/trace --demo           # or: an example parallel schedule
```

Then open `viz/index.html` in your browser (double-click it, no server
needed). It reads the timelines from `viz/trace.js`, which these commands
write; run one again and reload the page to see your latest code. Pick a
problem at the top and press **▶ Play**:

- **Mesh**: the cores on the grid, coloured by what they do at each moment
  (computing, receiving a DMA, waiting for a signal, idle, done). Arrows
  show data moving; tick **All DMAs** to see every transfer at once.
- **Status**: compute and DMA cycles of each core, and its utilization.
- **Per-core timeline**: one row per core; click anywhere to jump there.
  Gaps (idle or waiting) are the time you can still win.

Every upload on the web page shows the same timeline for each public problem.

## What to Modify

Edit **`solution.cpp`** only. Implement `solve()` to emit a better instruction
sequence than the baseline. The baseline runs everything on core 0 — you can
definitely beat it.

**`solve()` must work for any problem**, not only the six above: the final
round runs your code on hidden problems you have not seen. They use the same
operations and limits, but may have other grid sizes (always square, up to
4x4), other graphs, other DMA factors, inputs at any core and address, and
the output on any core. Read everything from `prob` (grid, `dma_factor`,
`ops`, `initial_tensors`, `output_core`, `output_addr`) and test with
`./build/check --random 50`.

## Scoring

- Each test case is **pass/fail on correctness** (output must match the golden
  reference within 0.1% relative tolerance).
- More correct problems ranks first; among equals, **lower total makespan
  wins**.
- The results also show the **compute utilization**: compute cycles of all
  cores / (cores × makespan). It does not count for the ranking; it tells you
  how much of the chip your schedule keeps busy.

## Submitting

Upload **`solution.cpp`** on the submission page,
**<https://hackathon.public.smd.dev>**. The first time, choose
**First submission** and type your team name: you get a **team code** (like
`ABCD-2345`). Write it down; you need it for every new version and to see
your submissions from another computer (the browser remembers it). The
server compiles your code (`g++ -std=c++17 -O2` with `hackathon.h`), runs `solve()`
on the public problems and simulates the instructions it emits. You see the
result of every problem, what `solve()` printed, and a timeline of what each
core did.

Every upload also runs on the **hidden problems** and tells you only how
many of them pass (for example 2 / 3). You do not see which ones, their
names, makespans, errors or what `solve()` printed.

Your **best correct makespan of every public problem** counts on the live
leaderboard, across all your submissions. It ranks teams by solved problems,
then by the sum of those best makespans.

A problem fails when your code:
- does not compile, or `solve()` crashes or runs out of time,
- uses an address or size outside the 4096-float SRAM,
- fires the same `signal_id` more than once,
- deadlocks (a `wait` whose signal is never fired),
- leaves a wrong result at `output_core` / `output_addr`.

## Final Round

When submissions close, the server runs **one submission of each team** on the
hidden problems: the one marked **Final** on the submission page (click
*Use for final* on any submission that compiled). If you do not choose, your
latest submission that compiled is used. The final ranking (solved hidden
problems, then total makespan) is published on the leaderboard afterwards.

## Rules

- Only modify `solution.cpp` (one file; it is all the server gets).
- No reading files, no network, no threads — just emit instructions.
- All data lives in SRAM. No global/shared memory.

## Tips

1. **Start with the baseline.** It works. Then improve one problem at a time.
2. **Problem 2 is the easiest win:** the two matmuls are independent — put them
   on different cores.
3. **For problem 3:** row-split the 32×32 matmul across 2 or 4 cores. Each
   core computes a strip of the output.
4. **Mind your SRAM budget.** 4096 floats total per core — plan your address
   layout on paper first.
5. **DMA is expensive at long distances**, and more when `prob->dma_factor` is
   high. Place ops on cores near their input data; `dma_cost()` tells you the
   exact cost of a copy.
6. **Overlap DMA with compute on _other_ cores.** While core 0 computes, core 1
   can be DMA-ing data it will need next.
