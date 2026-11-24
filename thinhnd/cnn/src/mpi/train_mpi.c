/*
 * Distributed training with MPI (data parallelism), following sections 5.1-5.2 of
 * Hegde & Usmani, "Parallel and Distributed Deep Learning".
 *
 *   --mode sync   every mini-batch is split over the ranks; gradients are summed along a
 *                 binomial tree and sent back. Same mathematics as sequential SGD.
 *   --mode avg    ParallelSGD: each rank runs SGD on its shard of a fresh global shuffle,
 *                 parameters are averaged once per epoch.
 *   --mode async  Downpour-style SGD: rank 0 is the parameter server, the other ranks push
 *                 gradients and receive new parameters without waiting for each other.
 *
 * Only point-to-point messages are used for the algorithms; the tree broadcast and the
 * tree reduction are implemented here.
 */
#include "../common/train_common.h"
#include <mpi.h>

#define MPI_REAL_T (sizeof(real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE)

enum { TAG_BCAST = 1, TAG_REDUCE, TAG_WORK, TAG_GRADIENT };

static int rank, size;
static double comm_seconds;

/* Binomial tree, ceil(log2 p) rounds: a rank receives from its parent, then serves its children. */
static void tree_bcast(void *buf, int count, MPI_Datatype type)
{
    const double begin = MPI_Wtime();
    int mask = 1;
    while (mask < size) {
        if (rank & mask) {
            MPI_Recv(buf, count, type, rank - mask, TAG_BCAST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        }
        mask <<= 1;
    }
    for (mask >>= 1; mask > 0; mask >>= 1)
        if (rank + mask < size)
            MPI_Send(buf, count, type, rank + mask, TAG_BCAST, MPI_COMM_WORLD);
    comm_seconds += MPI_Wtime() - begin;
}

/* The same tree upwards; the sum ends at rank 0. */
static void tree_sum(real *buf, real *tmp, int count)
{
    const double begin = MPI_Wtime();
    for (int mask = 1; mask < size; mask <<= 1) {
        if (rank & mask) {
            MPI_Send(buf, count, MPI_REAL_T, rank - mask, TAG_REDUCE, MPI_COMM_WORLD);
            break;
        }
        if (rank + mask < size) {
            MPI_Recv(tmp, count, MPI_REAL_T, rank + mask, TAG_REDUCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < count; i++) buf[i] += tmp[i];
        }
    }
    comm_seconds += MPI_Wtime() - begin;
}

typedef struct {
    Options o;
    Dataset d;
    real *w, *v, *g, *buf, *tmp;
    int *perm;
} State;

static double epoch_sync(State *s)
{
    const Options *o = &s->o;
    double loss = 0;
    for (int at = 0; at < o->train; at += o->batch) {
        const int m = o->train - at < o->batch ? o->train - at : o->batch;
        const int first = part_begin(m, size, rank), last = part_begin(m, size, rank + 1);

        s->g[NPARAM] = batch_gradient(s->w, &s->d, s->perm + at + first, last - first, s->g);
        tree_sum(s->g, s->tmp, NPARAM + 1);      /* all-to-one */
        tree_bcast(s->g, NPARAM + 1, MPI_REAL_T); /* one-to-all */
        sgd_update(s->w, s->v, s->g, (real)1 / m, o->lr, o->momentum);
        loss += s->g[NPARAM];
    }
    return loss / o->train;
}

static double epoch_average(State *s)
{
    const Options *o = &s->o;
    const int first = part_begin(o->train, size, rank), last = part_begin(o->train, size, rank + 1);
    real loss = 0;
    for (int at = first; at < last; at += o->batch) {
        const int m = last - at < o->batch ? last - at : o->batch;
        loss += batch_gradient(s->w, &s->d, s->perm + at, m, s->g);
        sgd_update(s->w, s->v, s->g, (real)1 / m, o->lr, o->momentum);
    }
    memcpy(s->buf, s->w, sizeof(real) * NPARAM);
    memcpy(s->buf + NPARAM, s->v, sizeof(real) * NPARAM);
    s->buf[2 * NPARAM] = loss;
    tree_sum(s->buf, s->tmp, 2 * NPARAM + 1);
    tree_bcast(s->buf, 2 * NPARAM + 1, MPI_REAL_T);
    for (int i = 0; i < NPARAM; i++) {
        s->w[i] = s->buf[i] / size;
        s->v[i] = s->buf[NPARAM + i] / size;
    }
    return s->buf[2 * NPARAM] / o->train;
}

/* Message to a worker: parameters + batch number (-1 = epoch finished). */
static double async_server(State *s)
{
    const Options *o = &s->o;
    const int batches = (o->train + o->batch - 1) / o->batch;
    /* stale gradients add up, so the step is divided by the number of workers */
    const real lr = o->lr / (real)sqrt(size - 1);
    int next = 0, active = 0;
    double loss = 0;

    for (int worker = 1; worker < size; worker++) {
        memcpy(s->buf, s->w, sizeof(real) * NPARAM);
        s->buf[NPARAM] = next < batches ? (real)next++ : (real)-1;
        active += s->buf[NPARAM] >= 0;
        MPI_Send(s->buf, NPARAM + 1, MPI_REAL_T, worker, TAG_WORK, MPI_COMM_WORLD);
    }
    while (active > 0) {
        MPI_Status status;
        MPI_Recv(s->g, NPARAM + 2, MPI_REAL_T, MPI_ANY_SOURCE, TAG_GRADIENT, MPI_COMM_WORLD, &status);
        sgd_update(s->w, s->v, s->g, (real)1 / s->g[NPARAM + 1], lr, o->momentum);
        loss += s->g[NPARAM];

        memcpy(s->buf, s->w, sizeof(real) * NPARAM);
        s->buf[NPARAM] = next < batches ? (real)next++ : (real)-1;
        active -= s->buf[NPARAM] < 0;
        MPI_Send(s->buf, NPARAM + 1, MPI_REAL_T, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
    }
    return loss / o->train;
}

static void async_worker(State *s)
{
    const Options *o = &s->o;
    for (;;) {
        double begin = MPI_Wtime();
        MPI_Recv(s->buf, NPARAM + 1, MPI_REAL_T, 0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        comm_seconds += MPI_Wtime() - begin;

        const int id = (int)s->buf[NPARAM];
        if (id < 0) return;
        const int at = id * o->batch, m = o->train - at < o->batch ? o->train - at : o->batch;
        s->g[NPARAM] = batch_gradient(s->buf, &s->d, s->perm + at, m, s->g);
        s->g[NPARAM + 1] = (real)m;

        begin = MPI_Wtime();
        MPI_Send(s->g, NPARAM + 2, MPI_REAL_T, 0, TAG_GRADIENT, MPI_COMM_WORLD);
        comm_seconds += MPI_Wtime() - begin;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    State s;
    s.o = parse_options(argc, argv);
    const int sync = !strcmp(s.o.mode, "sync");
    const int average = !strcmp(s.o.mode, "avg");
    const int async = !strcmp(s.o.mode, "async");
    if (!sync && !average && !async) die("--mode must be sync, avg or async", NULL);
    if (s.o.delay) die("--delay is only supported by train_cuda", NULL);
    if (async && size < 2) die("async needs at least 2 ranks", NULL);

    s.d = dataset_alloc();
    s.w = xalloc(NPARAM, sizeof(real));
    s.v = xalloc(NPARAM, sizeof(real));
    s.g = xalloc(NPARAM + 2, sizeof(real));
    s.buf = xalloc(2 * NPARAM + 1, sizeof(real));
    s.tmp = xalloc(2 * NPARAM + 1, sizeof(real));
    s.perm = xalloc(s.o.train, sizeof(int));

    /* rank 0 owns the files and the initial model */
    if (rank == 0) {
        dataset_load(s.o.data, &s.d);
        model_init(s.w, s.o.seed);
        printf("model params=%d train=%d test=%d batch=%d mode=%s ranks=%d\n",
               (int)NPARAM, s.o.train, (int)TEST_IMAGES, s.o.batch, s.o.mode, size);
    }
    tree_bcast(s.d.x, FILE_IMAGES * PIXELS, MPI_REAL_T);
    tree_bcast(s.d.y, FILE_IMAGES, MPI_INT);
    tree_bcast(s.w, NPARAM, MPI_REAL_T);
    comm_seconds = 0; /* initial distribution is not training time */

    double total = 0, loss = 0, val = 0, test = 0;
    for (int e = 1; e <= s.o.epochs; e++) {
        MPI_Barrier(MPI_COMM_WORLD); /* rank 0 may still have been evaluating */
        const double begin = MPI_Wtime();
        make_permutation(s.perm, s.o.train, s.o.seed, e);
        if (sync) loss = epoch_sync(&s);
        else if (average) loss = epoch_average(&s);
        else if (rank == 0) loss = async_server(&s);
        else async_worker(&s);

        double seconds = MPI_Wtime() - begin, slowest;
        MPI_Reduce(&seconds, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            total += slowest;
            val = accuracy(s.w, &s.d, VAL_BEGIN, VAL_IMAGES);
            test = accuracy(s.w, &s.d, TEST_BEGIN, TEST_IMAGES);
            print_epoch(e, loss, val, test, slowest);
        }
    }

    /* average time per rank spent communicating or waiting for other ranks */
    double comm_sum;
    MPI_Reduce(&comm_seconds, &comm_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("summary time=%.4f comm=%.4f loss=%.4f val=%.2f test=%.2f\n",
               total, comm_sum / (async ? size - 1 : size), loss, val, test);

    dataset_free(&s.d);
    free(s.w); free(s.v); free(s.g); free(s.buf); free(s.tmp); free(s.perm);
    MPI_Finalize();
    return 0;
}
