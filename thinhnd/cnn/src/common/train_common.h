/* Options, data loading and evaluation shared by the three training programs. */
#ifndef TRAIN_COMMON_H
#define TRAIN_COMMON_H

#include "model.h"
#include "util.h"

enum {
    FILE_IMAGES = 10000,
    MAX_TRAIN = 7000,                 /* images 0..6999 */
    VAL_IMAGES = 1000,                /* images 7000..7999: model and hyper-parameter selection */
    VAL_BEGIN = MAX_TRAIN,
    TEST_BEGIN = VAL_BEGIN + VAL_IMAGES,
    TEST_IMAGES = FILE_IMAGES - TEST_BEGIN, /* images 8000..9999: reported only */
    MAX_DELAY = 64
};

typedef struct {
    const char *data, *mode;
    int epochs, batch, train, seed;
    real lr, momentum;
    int bench, check; /* used by the CUDA program */
    int delay;        /* CUDA only: apply each gradient `delay` steps late */
} Options;

typedef struct {
    real *x;
    int *y;
} Dataset;

static inline Options parse_options(int argc, char **argv)
{
    Options o = {"data", "sync", 3, 64, MAX_TRAIN, 5408, (real)0.02, (real)0.9, 0, 0, 0};
    int lr_given = 0, momentum_given = 0;
    process_setup();
    for (int i = 1; i < argc; i++) {
        const char *key = argv[i];
        if (!strcmp(key, "--bench")) { o.bench = 1; continue; }
        if (!strcmp(key, "--check")) { o.check = 1; continue; }
        if (i + 1 == argc) die("missing value or unknown option:", key);
        const char *v = argv[++i];
        if (!strcmp(key, "--data")) o.data = v;
        else if (!strcmp(key, "--mode")) o.mode = v;
        else if (!strcmp(key, "--epochs")) o.epochs = atoi(v);
        else if (!strcmp(key, "--batch")) o.batch = atoi(v);
        else if (!strcmp(key, "--train")) o.train = atoi(v);
        else if (!strcmp(key, "--seed")) o.seed = atoi(v);
        else if (!strcmp(key, "--delay")) o.delay = atoi(v);
        else if (!strcmp(key, "--lr")) { o.lr = (real)atof(v); lr_given = 1; }
        else if (!strcmp(key, "--momentum")) { o.momentum = (real)atof(v); momentum_given = 1; }
        else die("unknown option:", key);
    }
    /* Momentum diverges with stale gradients, so Downpour uses plain SGD and a larger step. */
    if (!strcmp(o.mode, "async")) {
        if (!lr_given) o.lr = (real)0.1;
        if (!momentum_given) o.momentum = 0;
    }
    if (o.epochs < 1 || o.epochs > 1000) die("--epochs must be in [1, 1000]", NULL);
    if (o.batch < 1 || o.batch > 1024) die("--batch must be in [1, 1024]", NULL);
    if (o.train < o.batch || o.train > MAX_TRAIN) die("--train must be in [batch, 7000]", NULL);
    if (o.delay < 0 || o.delay > MAX_DELAY) die("--delay must be in [0, 64]", NULL);
    if (!(o.lr > 0) || o.momentum < 0 || o.momentum >= 1) die("bad --lr or --momentum", NULL);
    return o;
}

static inline Dataset dataset_alloc(void)
{
    Dataset d = {xalloc((size_t)FILE_IMAGES * PIXELS, sizeof(real)), xalloc(FILE_IMAGES, sizeof(int))};
    return d;
}

static inline void dataset_free(Dataset *d)
{
    free(d->x);
    free(d->y);
}

static inline void read_file(const char *dir, const char *name, unsigned char *buf, size_t size)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open", path);
    if (fread(buf, 1, size, f) != size) die("file is too short:", path);
    fclose(f);
}

/* Fashion-MNIST IDX files; the split is defined by the enum at the top. */
static inline void dataset_load(const char *dir, Dataset *d)
{
    const size_t image_bytes = 16 + (size_t)FILE_IMAGES * PIXELS, label_bytes = 8 + FILE_IMAGES;
    unsigned char *img = xalloc(image_bytes, 1), *lab = xalloc(label_bytes, 1);

    read_file(dir, "t10k-images-idx3-ubyte", img, image_bytes);
    read_file(dir, "t10k-labels-idx1-ubyte", lab, label_bytes);
    if (img[2] != 8 || img[3] != 3 || lab[2] != 8 || lab[3] != 1) die("not an IDX file in", dir);
    for (int i = 0; i < FILE_IMAGES * PIXELS; i++)
        d->x[i] = (real)((img[16 + i] / 255.0 - 0.2860) / 0.3530);
    for (int i = 0; i < FILE_IMAGES; i++) {
        d->y[i] = lab[8 + i];
        if (d->y[i] >= CLASSES) die("label out of range in", dir);
    }
    free(img);
    free(lab);
}

static inline real batch_gradient(const real *w, const Dataset *d, const int *idx, int m, real *g)
{
    real loss = 0;
    memset(g, 0, sizeof(real) * NPARAM);
    for (int i = 0; i < m; i++) loss += model_gradient(w, d->x + idx[i] * PIXELS, d->y[idx[i]], g);
    return loss;
}

static inline double accuracy(const real *w, const Dataset *d, int first, int count)
{
    int good = 0;
    for (int i = first; i < first + count; i++) good += model_predict(w, d->x + i * PIXELS) == d->y[i];
    return 100.0 * good / count;
}

static inline void print_epoch(int epoch, double loss, double val, double test, double seconds)
{
    printf("epoch=%d loss=%.4f val=%.2f test=%.2f time=%.4f\n", epoch, loss, val, test, seconds);
    fflush(stdout);
}

#endif
