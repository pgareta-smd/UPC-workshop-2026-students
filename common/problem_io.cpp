/*
 * problem_io.cpp — Problems described in a text file, and random problems.
 */
#include "problem_io.h"
#include <cctype>

OwnedProblem::~OwnedProblem()
{
    for (int i = 0; i < p.num_initial_tensors; i++)
        std::free(p.initial_tensors[i].data);
    std::free(p.expected_output);
}

/** Same generator as make_data() in hackathon.cpp. */
static float *make_data(int size, unsigned seed)
{
    float *d = (float *)std::calloc(size, sizeof(float));
    unsigned s = seed;
    for (int i = 0; i < size; i++) {
        s = s * 1103515245u + 12345u;
        d[i] = (float)((s >> 16) & 0xFF) / 255.0f + 0.01f;
    }
    return d;
}

/* ---------------------------------------------------------------- */
/* Building and checking a problem                                   */
/* ---------------------------------------------------------------- */

/** Adds ops one by one; every method returns an error or "". */
struct Builder {
    Problem *p;
    bool has_grid = false, has_output = false, has_factor = false;

    std::string grid(long rows, long cols)
    {
        if (has_grid) return "grid given twice";
        if (rows < 1 || cols < 1 || rows * cols > MAX_CORES)
            return "grid must have 1 to 16 cores";
        if (rows != cols) return "grid must be square";
        p->grid_rows = (int)rows;
        p->grid_cols = (int)cols;
        p->num_cores = (int)(rows * cols);
        p->dma_factor = 1.0;
        has_grid = true;
        return "";
    }

    std::string dma_factor(double f)
    {
        if (!has_grid) return "grid must come first";
        if (has_factor) return "dma_factor given twice";
        if (!(f >= 0.1 && f <= 16)) return "dma_factor must be 0.1 to 16";
        p->dma_factor = f;
        has_factor = true;
        return "";
    }

    std::string new_op(long id, GraphOpType type, GraphOp **op)
    {
        if (has_output) return "op after output";
        if (id != p->num_ops) return "ops must be numbered 0, 1, 2... in order";
        if (p->num_ops >= MAX_GRAPH_OPS) return "too many ops";
        *op = &p->ops[p->num_ops++];
        std::memset(*op, 0, sizeof(GraphOp));
        (*op)->id = (int)id;
        (*op)->type = type;
        return "";
    }

    /** Size of an earlier op, or -1 if @p a is not one. */
    long size_of(long a, long id)
    {
        return a >= 0 && a < id ? p->ops[a].output_size : -1;
    }

    std::string input(long id, long size, long core, long addr, long seed)
    {
        if (!has_grid) return "grid must come first";
        if (size < 1 || size > SRAM_FLOATS) return "bad input size";
        if (core < 0 || core >= p->num_cores) return "input core out of grid";
        if (addr < 0 || addr + size > SRAM_FLOATS) return "input outside SRAM";
        for (int t = 0; t < p->num_initial_tensors; t++) {
            const InitialTensor &o = p->initial_tensors[t];
            if (o.core_id == core && addr < o.sram_addr + o.num_floats &&
                o.sram_addr < addr + size)
                return "inputs overlap";
        }
        GraphOp *op;
        std::string e = new_op(id, GRAPH_INPUT, &op);
        if (!e.empty()) return e;
        op->output_size = (int)size;
        InitialTensor &t = p->initial_tensors[p->num_initial_tensors++];
        t.op_id = (int)id;
        t.core_id = (int)core;
        t.sram_addr = (int)addr;
        t.num_floats = (int)size;
        t.data = make_data((int)size, (unsigned)seed);
        return "";
    }

    std::string matmul(long id, long a, long b, long M, long K, long N)
    {
        if (M < 1 || K < 1 || N < 1 || M * N > SRAM_FLOATS)
            return "bad matmul dims";
        if (size_of(a, id) != M * K || size_of(b, id) != K * N)
            return "matmul inputs do not match M, K, N";
        GraphOp *op;
        std::string e = new_op(id, GRAPH_MATMUL, &op);
        if (!e.empty()) return e;
        op->M = (int)M; op->K = (int)K; op->N = (int)N;
        op->output_size = (int)(M * N);
        op->num_inputs = 2;
        op->input_ids[0] = (int)a;
        op->input_ids[1] = (int)b;
        return "";
    }

    std::string unary(long id, GraphOpType type, long a)
    {
        long sa = size_of(a, id);
        if (sa < 0) return "unknown input op";
        GraphOp *op;
        std::string e = new_op(id, type, &op);
        if (!e.empty()) return e;
        op->output_size = (int)sa;
        op->num_inputs = 1;
        op->input_ids[0] = (int)a;
        return "";
    }

    std::string binary(long id, GraphOpType type, long a, long b)
    {
        long sa = size_of(a, id), sb = size_of(b, id);
        if (sa < 0 || sb < 0) return "unknown input op";
        if (type == GRAPH_ADD && sa != sb) return "add inputs differ in size";
        long size = type == GRAPH_ADD ? sa : sa + sb;
        if (size > SRAM_FLOATS) return "concat larger than SRAM";
        GraphOp *op;
        std::string e = new_op(id, type, &op);
        if (!e.empty()) return e;
        op->output_size = (int)size;
        op->num_inputs = 2;
        op->input_ids[0] = (int)a;
        op->input_ids[1] = (int)b;
        return "";
    }

    std::string output(long id, long core, long addr)
    {
        if (has_output) return "output given twice";
        if (id < 0 || id >= p->num_ops) return "unknown output op";
        if (core < 0 || core >= p->num_cores) return "output core out of grid";
        long size = p->ops[id].output_size;
        if (addr < 0 || addr + size > SRAM_FLOATS) return "output outside SRAM";
        p->output_op_id = (int)id;
        p->output_core = (int)core;
        p->output_addr = (int)addr;
        p->output_size = (int)size;
        has_output = true;
        return "";
    }
};

/* ---------------------------------------------------------------- */
/* Reference result (same loops as hackathon.cpp)                    */
/* ---------------------------------------------------------------- */

void compute_expected(Problem *p)
{
    std::vector<std::vector<float>> out(p->num_ops);
    for (int t = 0; t < p->num_initial_tensors; t++) {
        const InitialTensor &it = p->initial_tensors[t];
        out[it.op_id].assign(it.data, it.data + it.num_floats);
    }
    for (int i = 0; i < p->num_ops; i++) {
        const GraphOp &op = p->ops[i];
        if (op.type == GRAPH_INPUT) continue;
        std::vector<float> &o = out[i];
        o.assign(op.output_size, 0.0f);
        const std::vector<float> &a = out[op.input_ids[0]];
        switch (op.type) {
        case GRAPH_MATMUL: {
            const std::vector<float> &b = out[op.input_ids[1]];
            for (int r = 0; r < op.M; r++)
                for (int c = 0; c < op.N; c++) {
                    float sum = 0;
                    for (int k = 0; k < op.K; k++)
                        sum += a[r * op.K + k] * b[k * op.N + c];
                    o[r * op.N + c] = sum;
                }
            break;
        }
        case GRAPH_RELU:
            for (int k = 0; k < op.output_size; k++) o[k] = a[k] > 0 ? a[k] : 0;
            break;
        case GRAPH_ADD: {
            const std::vector<float> &b = out[op.input_ids[1]];
            for (int k = 0; k < op.output_size; k++) o[k] = a[k] + b[k];
            break;
        }
        case GRAPH_CONCAT: {
            const std::vector<float> &b = out[op.input_ids[1]];
            std::copy(a.begin(), a.end(), o.begin());
            std::copy(b.begin(), b.end(), o.begin() + a.size());
            break;
        }
        default: break;
        }
    }
    std::free(p->expected_output);
    p->expected_output = (float *)std::calloc(p->output_size, sizeof(float));
    const std::vector<float> &res = out[p->output_op_id];
    std::copy(res.begin(), res.end(), p->expected_output);
}

/* ---------------------------------------------------------------- */
/* Parsing                                                           */
/* ---------------------------------------------------------------- */

static bool read_longs(FILE *f, long *v, int n)
{
    for (int i = 0; i < n; i++) {
        long long x;
        if (std::fscanf(f, "%lld", &x) != 1 || x < -1000000000LL ||
            x > 1000000000LL)
            return false;
        v[i] = (long)x;
    }
    return true;
}

/** Reads the next word, skipping '#' comments; counts lines. */
static bool next_word(FILE *f, char *w, size_t n, long *line)
{
    int c;
    for (;;) {
        c = std::fgetc(f);
        if (c == EOF) return false;
        if (c == '\n') (*line)++;
        else if (c == '#') {
            while ((c = std::fgetc(f)) != EOF && c != '\n') {}
            if (c == EOF) return false;
            (*line)++;
        } else if (!std::isspace(c)) break;
    }
    size_t i = 0;
    while (c != EOF && !std::isspace(c)) {
        if (i + 1 < n) w[i++] = (char)c;
        c = std::fgetc(f);
    }
    w[i] = 0;
    if (c != EOF) std::ungetc(c, f);
    return true;
}

static bool valid_name(const char *s)
{
    if (!*s) return false;
    for (; *s; s++)
        if (!std::isalnum((unsigned char)*s) && *s != '_' && *s != '-')
            return false;
    return true;
}

std::string read_problems(FILE *f, ProblemList *out)
{
    char w[80];
    long line = 1;
    std::unique_ptr<OwnedProblem> cur;
    Builder b{nullptr};
    auto fail = [&](const std::string &why) {
        return "line " + std::to_string(line) + ": " + why;
    };
    while (next_word(f, w, sizeof(w), &line)) {
        long v[6];
        std::string e;
        if (!std::strcmp(w, "problem")) {
            if (cur) return fail("missing 'end'");
            if (!read_longs(f, v, 1) || v[0] < 0 || v[0] > 999)
                return fail("bad problem id");
            for (auto &q : *out)
                if (q->id == v[0]) return fail("problem id used twice");
            if (!next_word(f, w, sizeof(w), &line) || !valid_name(w) ||
                std::strlen(w) >= sizeof(cur->p.name))
                return fail("bad problem name");
            cur.reset(new OwnedProblem());
            cur->id = (int)v[0];
            std::strcpy(cur->p.name, w);
            b = Builder{&cur->p};
            continue;
        }
        if (!cur) return fail("expected 'problem'");
        if (!std::strcmp(w, "grid")) {
            e = read_longs(f, v, 2) ? b.grid(v[0], v[1]) : "bad numbers";
        } else if (!std::strcmp(w, "dma_factor")) {
            double x;
            e = std::fscanf(f, "%lf", &x) == 1 ? b.dma_factor(x) : "bad number";
        } else if (!std::strcmp(w, "input")) {
            e = read_longs(f, v, 5) ? b.input(v[0], v[1], v[2], v[3], v[4])
                                    : "bad numbers";
        } else if (!std::strcmp(w, "matmul")) {
            e = read_longs(f, v, 6) ? b.matmul(v[0], v[1], v[2], v[3], v[4], v[5])
                                    : "bad numbers";
        } else if (!std::strcmp(w, "relu")) {
            e = read_longs(f, v, 2) ? b.unary(v[0], GRAPH_RELU, v[1])
                                    : "bad numbers";
        } else if (!std::strcmp(w, "add") || !std::strcmp(w, "concat")) {
            GraphOpType t = w[0] == 'a' ? GRAPH_ADD : GRAPH_CONCAT;
            e = read_longs(f, v, 3) ? b.binary(v[0], t, v[1], v[2])
                                    : "bad numbers";
        } else if (!std::strcmp(w, "output")) {
            e = read_longs(f, v, 3) ? b.output(v[0], v[1], v[2]) : "bad numbers";
        } else if (!std::strcmp(w, "end")) {
            if (!b.has_output) return fail("problem without output");
            compute_expected(&cur->p);
            out->push_back(std::move(cur));
            continue;
        } else {
            return fail(std::string("unknown keyword '") + w + "'");
        }
        if (!e.empty()) return fail(e);
    }
    if (cur) return fail("missing 'end'");
    return "";
}

ProblemList load_problem_file(const char *path)
{
    FILE *f = std::fopen(path, "r");
    if (!f) { std::perror(path); std::exit(1); }
    ProblemList list;
    std::string err = read_problems(f, &list);
    std::fclose(f);
    if (!err.empty()) {
        std::fprintf(stderr, "%s: %s\n", path, err.c_str());
        std::exit(1);
    }
    return list;
}

/* ---------------------------------------------------------------- */
/* Random problems                                                   */
/* ---------------------------------------------------------------- */

namespace {

struct Rng {
    unsigned long long s;
    unsigned next()
    {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (unsigned)(s >> 11);
    }
    int pick(std::initializer_list<int> v)
    {
        return *(v.begin() + next() % v.size());
    }
    bool chance(int percent) { return (int)(next() % 100) < percent; }
};

/** One generation attempt; returns false if it did not fit. */
bool try_random(Rng &rng, OwnedProblem *out)
{
    Problem *p = &out->p;
    Builder b{p};
    int n = rng.pick({2, 3, 4});
    b.grid(n, n);

    /* inputs go to the next free address of a random core */
    int cursor[MAX_CORES] = {0};
    long total = 0;
    bool ok = true;
    auto check = [&](const std::string &e) { if (!e.empty()) ok = false; };
    auto input = [&](int size) {
        int core = rng.next() % p->num_cores;
        int id = p->num_ops;
        check(b.input(id, size, core, cursor[core], rng.next() % 1000));
        cursor[core] += size;
        total += size;
        return id;
    };
    auto add_op = [&](long size) { total += size; return p->num_ops; };

    int rows = rng.pick({4, 8, 16}), cols = rng.pick({8, 16, 32});
    int cur = input(rows * cols);
    int layers = 1 + rng.next() % 4;
    for (int l = 0; l < layers && ok; l++) {
        int N = rng.pick({4, 8, 16, 32});
        if (rng.chance(30)) {
            /* two branches joined by a concat, which stacks the rows */
            int w1 = input(cols * N);
            int y1 = add_op(rows * N);
            check(b.matmul(y1, cur, w1, rows, cols, N));
            int w2 = input(cols * N);
            int y2 = add_op(rows * N);
            check(b.matmul(y2, cur, w2, rows, cols, N));
            cur = add_op(2L * rows * N);
            check(b.binary(cur, GRAPH_CONCAT, y1, y2));
            rows *= 2;
        } else {
            int w = input(cols * N);
            int y = add_op(rows * N);
            check(b.matmul(y, cur, w, rows, cols, N));
            cur = y;
        }
        cols = N;
        if (ok && rng.chance(50)) {
            int bias = input(rows * cols);
            int y = add_op((long)rows * cols);
            check(b.binary(y, GRAPH_ADD, cur, bias));
            cur = y;
        }
        if (ok && rng.chance(60)) {
            int y = add_op((long)rows * cols);
            check(b.unary(y, GRAPH_RELU, cur));
            cur = y;
        }
        if (total > 3400) ok = false;
    }
    if (!ok) return false;
    /* output at the top of the SRAM of a random core */
    int size = p->ops[cur].output_size;
    if (total + size > SRAM_FLOATS) return false;
    if (!b.output(cur, rng.next() % p->num_cores, SRAM_FLOATS - size).empty())
        return false;
    b.dma_factor(rng.pick({10, 10, 15, 20, 30}) / 10.0);
    return true;
}

} // namespace

void random_problem(unsigned seed, OwnedProblem *out)
{
    Rng rng{0x9E3779B97F4A7C15ULL ^ ((unsigned long long)seed * 0x100000001B3ULL)};
    for (;;) {
        out->~OwnedProblem();
        new (out) OwnedProblem();
        out->id = (int)seed;
        if (try_random(rng, out)) break;
    }
    std::snprintf(out->p.name, sizeof(out->p.name), "random_%u", seed);
    compute_expected(&out->p);
}
