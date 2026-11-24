/* Sequential mini-batch SGD: the baseline for every speedup. */
#include "../common/train_common.h"

int main(int argc, char **argv)
{
    Options o = parse_options(argc, argv);
    Dataset d = dataset_alloc();
    real *w = xalloc(NPARAM, sizeof(real)), *v = xalloc(NPARAM, sizeof(real)), *g = xalloc(NPARAM, sizeof(real));
    int *perm = xalloc(o.train, sizeof(int));
    double total = 0, val = 0, test = 0, loss = 0;

    if (o.delay) die("--delay is only supported by train_cuda", NULL);
    dataset_load(o.data, &d);
    model_init(w, o.seed);
    printf("model params=%d train=%d test=%d batch=%d\n", (int)NPARAM, o.train, (int)TEST_IMAGES, o.batch);

    for (int e = 1; e <= o.epochs; e++) {
        const double begin = now_seconds();
        make_permutation(perm, o.train, o.seed, e);
        loss = 0;
        for (int at = 0; at < o.train; at += o.batch) {
            const int m = o.train - at < o.batch ? o.train - at : o.batch;
            loss += batch_gradient(w, &d, perm + at, m, g);
            sgd_update(w, v, g, (real)1 / m, o.lr, o.momentum);
        }
        const double seconds = now_seconds() - begin;
        total += seconds;
        loss /= o.train;
        val = accuracy(w, &d, VAL_BEGIN, VAL_IMAGES); /* not timed */
        test = accuracy(w, &d, TEST_BEGIN, TEST_IMAGES);
        print_epoch(e, loss, val, test, seconds);
    }
    printf("summary time=%.4f comm=0 loss=%.4f val=%.2f test=%.2f\n", total, loss, val, test);

    dataset_free(&d);
    free(w); free(v); free(g); free(perm);
    return 0;
}
