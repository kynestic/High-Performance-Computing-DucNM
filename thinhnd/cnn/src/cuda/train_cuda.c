/*
 * CUDA host program in C (driver API). kernels.cu is compiled with NVRTC, which ships with
 * the CUDA Toolkit, so neither nvcc nor Visual Studio is needed.
 *
 *   train_cuda            train the CNN
 *   train_cuda --check    compare the GPU gradient with the sequential C gradient
 *   train_cuda --bench    CPU vs GPU time of a convolution and of a dense layer
 */
#include "../common/train_common.h"
#include <cuda.h>
#include <nvrtc.h>

#define CHECK(call) cu_check((call), #call)
#define BLOCK 128
#define AT(ptr, count) ((ptr) + (CUdeviceptr)(count) * sizeof(float))

enum {
    K_LOAD, K_CONV, K_POOL, K_POOL_BACK, K_CONV_INPUT, K_CONV_WEIGHT, K_SUM,
    K_DENSE, K_DENSE_INPUT, K_DENSE_WEIGHT, K_SOFTMAX, K_SGD, K_COUNT
};

static const char *kernel_name[K_COUNT] = {
    "load_batch", "conv_forward", "pool_forward", "pool_relu_backward", "conv_backward_input",
    "conv_backward_weight", "sum_samples", "dense_forward", "dense_backward_input",
    "dense_backward_weight", "softmax", "sgd_update"
};

static CUfunction kernel[K_COUNT];

typedef struct {
    CUdeviceptr data, label, perm, w, v, g;
    CUdeviceptr x, a1, p1, at1, a2, p2, at2, h, logit; /* forward */
    CUdeviceptr df, dh, dp2, da2, dp1, da1, part;      /* backward */
    CUdeviceptr loss, correct;
} Gpu;

static void cu_check(CUresult r, const char *what)
{
    if (r == CUDA_SUCCESS) return;
    const char *text = NULL;
    cuGetErrorString(r, &text);
    fprintf(stderr, "CUDA call failed: %s\n", what);
    die("CUDA:", text);
}

static CUdeviceptr gpu_alloc(size_t count)
{
    CUdeviceptr p;
    CHECK(cuMemAlloc(&p, count * sizeof(float))); /* float and int have the same size */
    return p;
}

static void launch(int id, int threads, void **args)
{
    CHECK(cuLaunchKernel(kernel[id], (threads + BLOCK - 1) / BLOCK, 1, 1, BLOCK, 1, 1, 0, 0, args, NULL));
}

static CUmodule load_kernels(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("run from the project root; cannot open", path);
    fseek(f, 0, SEEK_END);
    const long length = ftell(f);
    rewind(f);
    char *source = xalloc(length + 1, 1);
    if (fread(source, 1, length, f) != (size_t)length) die("cannot read", path);
    fclose(f);

    nvrtcProgram program;
    if (nvrtcCreateProgram(&program, source, path, 0, NULL, NULL) != NVRTC_SUCCESS) die("NVRTC create failed", NULL);
    const nvrtcResult result = nvrtcCompileProgram(program, 0, NULL);
    size_t size;
    if (result != NVRTC_SUCCESS) {
        nvrtcGetProgramLogSize(program, &size);
        char *log = xalloc(size + 1, 1);
        nvrtcGetProgramLog(program, log);
        die("kernel compilation failed\n", log);
    }
    if (nvrtcGetPTXSize(program, &size) != NVRTC_SUCCESS) die("NVRTC PTX failed", NULL);
    char *ptx = xalloc(size + 1, 1);
    if (nvrtcGetPTX(program, ptx) != NVRTC_SUCCESS) die("NVRTC PTX failed", NULL);
    nvrtcDestroyProgram(&program);

    CUmodule module;
    CHECK(cuModuleLoadData(&module, ptx));
    for (int i = 0; i < K_COUNT; i++) CHECK(cuModuleGetFunction(&kernel[i], module, kernel_name[i]));
    free(source);
    free(ptx);
    return module;
}

/* ---------- one function per layer */

static void conv(CUdeviceptr in, CUdeviceptr out, CUdeviceptr w, int m, int cin, int cout, int side)
{
    int ks = KS;
    void *a[] = {&in, &out, &w, &m, &cin, &cout, &side, &ks};
    launch(K_CONV, m * cout * side * side, a);
}

static void pool(CUdeviceptr in, CUdeviceptr out, CUdeviceptr at, int m, int channels, int side)
{
    void *a[] = {&in, &out, &at, &m, &channels, &side};
    launch(K_POOL, m * channels * (side / 2) * (side / 2), a);
}

static void pool_back(CUdeviceptr dout, CUdeviceptr at, CUdeviceptr act, CUdeviceptr din, int m, int channels, int side)
{
    void *a[] = {&dout, &at, &act, &din, &m, &channels, &side};
    launch(K_POOL_BACK, m * channels * side * side, a);
}

static void conv_back(const Gpu *d, CUdeviceptr in, CUdeviceptr dout, CUdeviceptr din,
                      int offset, int m, int cin, int cout, int side)
{
    int ks = KS, count = cout * cin * KS * KS + cout;
    CUdeviceptr w = AT(d->w, offset), g = AT(d->g, offset), part = d->part;
    if (din) {
        void *a[] = {&dout, &din, &w, &m, &cin, &cout, &side, &ks};
        launch(K_CONV_INPUT, m * cin * side * side, a);
    }
    void *map[] = {&in, &dout, &part, &m, &cin, &cout, &side, &ks};
    launch(K_CONV_WEIGHT, m * count, map);
    void *reduce[] = {&part, &g, &m, &count};
    launch(K_SUM, count, reduce);
}

static void dense(CUdeviceptr in, CUdeviceptr out, CUdeviceptr w, int m, int nin, int nout, int relu)
{
    void *a[] = {&in, &out, &w, &m, &nin, &nout, &relu};
    launch(K_DENSE, m * nout, a);
}

static void dense_back(const Gpu *d, CUdeviceptr in, CUdeviceptr dout, CUdeviceptr act, CUdeviceptr din,
                       int offset, int m, int nin, int nout)
{
    CUdeviceptr w = AT(d->w, offset), g = AT(d->g, offset);
    void *a[] = {&dout, &act, &din, &w, &m, &nin, &nout};
    launch(K_DENSE_INPUT, m * nin, a);
    void *b[] = {&in, &dout, &act, &g, &m, &nin, &nout};
    launch(K_DENSE_WEIGHT, nout * nin + nout, b);
}

/* at: position of the batch inside the permutation */
static void forward(const Gpu *d, int at, int m)
{
    int pixels = PIXELS, classes = CLASSES;
    CUdeviceptr data = d->data, x = d->x, perm = AT(d->perm, at);
    void *load[] = {&data, &perm, &x, &m, &pixels};
    launch(K_LOAD, m * PIXELS, load);

    conv(d->x, d->a1, AT(d->w, P_CONV1), m, 1, C1, IMG);
    pool(d->a1, d->p1, d->at1, m, C1, IMG);
    conv(d->p1, d->a2, AT(d->w, P_CONV2), m, C1, C2, S1);
    pool(d->a2, d->p2, d->at2, m, C2, S1);
    dense(d->p2, d->h, AT(d->w, P_DENSE1), m, FLAT, HIDDEN, 1);
    dense(d->h, d->logit, AT(d->w, P_DENSE2), m, HIDDEN, CLASSES, 0);

    CUdeviceptr logit = d->logit, label = d->label, df = d->df;
    CUdeviceptr loss = AT(d->loss, at), correct = AT(d->correct, at);
    void *a[] = {&logit, &label, &perm, &df, &loss, &correct, &m, &classes};
    launch(K_SOFTMAX, m, a);
}

static void backward(const Gpu *d, int m)
{
    dense_back(d, d->h, d->df, 0, d->dh, P_DENSE2, m, HIDDEN, CLASSES);
    dense_back(d, d->p2, d->dh, d->h, d->dp2, P_DENSE1, m, FLAT, HIDDEN);
    pool_back(d->dp2, d->at2, d->a2, d->da2, m, C2, S1);
    conv_back(d, d->p1, d->da2, d->dp1, P_CONV2, m, C1, C2, S1);
    pool_back(d->dp1, d->at1, d->a1, d->da1, m, C1, IMG);
    conv_back(d, d->x, d->da1, 0, P_CONV1, m, 1, C1, IMG);
}

static void update(const Gpu *d, int m, float lr, float momentum)
{
    int count = NPARAM;
    float scale = 1.f / m;
    CUdeviceptr w = d->w, v = d->v, g = d->g;
    void *a[] = {&w, &v, &g, &count, &scale, &lr, &momentum};
    launch(K_SGD, NPARAM, a);
}

static Gpu gpu_create(const Dataset *data, const real *w, int cap)
{
    Gpu d;
    d.data = gpu_alloc((size_t)FILE_IMAGES * PIXELS);
    d.label = gpu_alloc(FILE_IMAGES);
    d.perm = gpu_alloc(FILE_IMAGES);
    d.loss = gpu_alloc(FILE_IMAGES);
    d.correct = gpu_alloc(FILE_IMAGES);
    d.w = gpu_alloc(NPARAM);
    d.v = gpu_alloc(NPARAM);
    d.g = gpu_alloc(NPARAM);
    d.x = gpu_alloc(cap * PIXELS);
    d.a1 = gpu_alloc(cap * C1 * PIXELS);
    d.da1 = gpu_alloc(cap * C1 * PIXELS);
    d.p1 = gpu_alloc(cap * C1 * S1 * S1);
    d.dp1 = gpu_alloc(cap * C1 * S1 * S1);
    d.at1 = gpu_alloc(cap * C1 * S1 * S1);
    d.a2 = gpu_alloc(cap * C2 * S1 * S1);
    d.da2 = gpu_alloc(cap * C2 * S1 * S1);
    d.p2 = gpu_alloc(cap * FLAT);
    d.dp2 = gpu_alloc(cap * FLAT);
    d.at2 = gpu_alloc(cap * FLAT);
    d.h = gpu_alloc(cap * HIDDEN);
    d.dh = gpu_alloc(cap * HIDDEN);
    d.logit = gpu_alloc(cap * CLASSES);
    d.df = gpu_alloc(cap * CLASSES);
    d.part = gpu_alloc(cap * (C2 * C1 * KS * KS + C2));
    CHECK(cuMemcpyHtoD(d.data, data->x, sizeof(float) * FILE_IMAGES * PIXELS));
    CHECK(cuMemcpyHtoD(d.label, data->y, sizeof(int) * FILE_IMAGES));
    CHECK(cuMemcpyHtoD(d.w, w, sizeof(float) * NPARAM));
    CHECK(cuMemsetD32(d.v, 0, NPARAM));
    return d;
}

static double gpu_accuracy(const Gpu *d, int batch, int first, int count, int *perm, int *good)
{
    for (int i = 0; i < count; i++) perm[i] = first + i;
    CHECK(cuMemcpyHtoD(d->perm, perm, sizeof(int) * count));
    for (int at = 0; at < count; at += batch) forward(d, at, count - at < batch ? count - at : batch);
    CHECK(cuMemcpyDtoH(good, d->correct, sizeof(int) * count));
    int sum = 0;
    for (int i = 0; i < count; i++) sum += good[i];
    return 100.0 * sum / count;
}

/*
 * With --delay D the gradient computed at step t from w_t is applied at step t + D: the
 * deterministic equivalent of D + 1 asynchronous workers served in round-robin order.
 */
static void train(const Options *o, Gpu *d)
{
    CUdeviceptr pending[MAX_DELAY + 1];
    int pending_m[MAX_DELAY + 1];
    const int slots = o->delay + 1;
    long step = 0;
    pending[0] = d->g;
    for (int i = 1; i < slots; i++) pending[i] = gpu_alloc(NPARAM);

    int *perm = xalloc(FILE_IMAGES, sizeof(int)), *good = xalloc(FILE_IMAGES, sizeof(int));
    float *losses = xalloc(FILE_IMAGES, sizeof(float));
    double total = 0, loss = 0, val = 0, test = 0;

    for (int e = 1; e <= o->epochs; e++) {
        const double begin = now_seconds();
        make_permutation(perm, o->train, o->seed, e);
        CHECK(cuMemcpyHtoD(d->perm, perm, sizeof(int) * o->train));
        for (int at = 0; at < o->train; at += o->batch) {
            const int m = o->train - at < o->batch ? o->train - at : o->batch;
            d->g = pending[step % slots];
            pending_m[step % slots] = m;
            forward(d, at, m);
            backward(d, m);
            if (step >= o->delay) {
                const long due = (step - o->delay) % slots;
                d->g = pending[due];
                update(d, pending_m[due], o->lr, o->momentum);
            }
            step++;
        }
        CHECK(cuCtxSynchronize());
        CHECK(cuMemcpyDtoH(losses, d->loss, sizeof(float) * o->train));
        const double seconds = now_seconds() - begin;
        total += seconds;

        loss = 0;
        for (int i = 0; i < o->train; i++) loss += losses[i];
        loss /= o->train;

        val = gpu_accuracy(d, o->batch, VAL_BEGIN, VAL_IMAGES, perm, good);
        test = gpu_accuracy(d, o->batch, TEST_BEGIN, TEST_IMAGES, perm, good);
        print_epoch(e, loss, val, test, seconds);
    }
    printf("summary time=%.4f comm=0 loss=%.4f val=%.2f test=%.2f\n", total, loss, val, test);
    free(perm); free(good); free(losses);
}

static int check_gradient(const Options *o, Gpu *d, const Dataset *data, const real *w)
{
    const int m = o->batch;
    int *perm = xalloc(o->train, sizeof(int));
    float *cpu = xalloc(NPARAM, sizeof(float)), *gpu = xalloc(NPARAM, sizeof(float));

    make_permutation(perm, o->train, o->seed, 1);
    CHECK(cuMemcpyHtoD(d->perm, perm, sizeof(int) * m));
    forward(d, 0, m);
    backward(d, m);
    CHECK(cuMemcpyDtoH(gpu, d->g, sizeof(float) * NPARAM));
    batch_gradient(w, data, perm, m, cpu);

    double diff = 0, norm = 0;
    for (int i = 0; i < NPARAM; i++) {
        diff += (double)(cpu[i] - gpu[i]) * (cpu[i] - gpu[i]);
        norm += (double)cpu[i] * cpu[i];
    }
    const double error = sqrt(diff / norm);
    printf("check relative_error=%.3e %s\n", error, error < 1e-4 ? "PASS" : "FAIL");
    free(perm); free(cpu); free(gpu);
    return error < 1e-4 ? 0 : 1;
}

/* Section 3 of the paper: time against batch size and against matrix size. */
static void benchmark(const Dataset *data, const real *w)
{
    static const int batches[] = {1, 4, 16, 64, 256, 1024}, sizes[] = {128, 256, 512, 1024, 2048, 4096};
    float *out = xalloc(1024 * C1 * PIXELS, sizeof(float));
    CUdeviceptr gx = gpu_alloc(1024 * PIXELS), gy = gpu_alloc(1024 * C1 * PIXELS), gw = gpu_alloc(P_CONV2);
    CHECK(cuMemcpyHtoD(gx, data->x, sizeof(float) * 1024 * PIXELS));
    CHECK(cuMemcpyHtoD(gw, w, sizeof(float) * P_CONV2));
    conv(gx, gy, gw, 64, 1, C1, IMG); /* warm-up */
    CHECK(cuCtxSynchronize());

    for (int k = 0; k < 6; k++) {
        const int m = batches[k], reps = 2048 / m < 100 ? (2048 / m < 10 ? 10 : 2048 / m) : 100;
        double t = now_seconds();
        for (int r = 0; r < reps; r++)
            for (int s = 0; s < m; s++)
                conv_forward(data->x + s * PIXELS, out + s * C1 * PIXELS, w, 1, C1, IMG);
        const double cpu = (now_seconds() - t) / reps;
        t = now_seconds();
        for (int r = 0; r < reps; r++) {
            conv(gx, gy, gw, m, 1, C1, IMG);
            CHECK(cuCtxSynchronize());
        }
        const double gpu = (now_seconds() - t) / reps;
        printf("bench layer=conv size=%d cpu=%.7f gpu=%.7f\n", m, cpu, gpu);
    }
    CHECK(cuMemFree(gx)); CHECK(cuMemFree(gy)); CHECK(cuMemFree(gw));

    for (int k = 0; k < 6; k++) {
        const int n = sizes[k], reps = n < 1024 ? 100 : 10;
        float *mat = xalloc((size_t)n * n + n, sizeof(float)), *x = xalloc(n, sizeof(float));
        uint64_t state = n;
        for (int i = 0; i < n * n + n; i++) mat[i] = (float)(rng_uniform(&state) - 0.5);
        for (int i = 0; i < n; i++) x[i] = (float)rng_uniform(&state);
        CUdeviceptr gm = gpu_alloc((size_t)n * n + n), gv = gpu_alloc(n), gr = gpu_alloc(n);
        CHECK(cuMemcpyHtoD(gm, mat, sizeof(float) * ((size_t)n * n + n)));
        CHECK(cuMemcpyHtoD(gv, x, sizeof(float) * n));
        dense(gv, gr, gm, 1, n, n, 0);
        CHECK(cuCtxSynchronize());

        double t = now_seconds();
        for (int r = 0; r < reps; r++) dense_forward(x, out, mat, n, n, 0);
        const double cpu = (now_seconds() - t) / reps;
        t = now_seconds();
        for (int r = 0; r < reps; r++) {
            dense(gv, gr, gm, 1, n, n, 0);
            CHECK(cuCtxSynchronize());
        }
        const double gpu = (now_seconds() - t) / reps;
        printf("bench layer=dense size=%d cpu=%.7f gpu=%.7f\n", n, cpu, gpu);
        CHECK(cuMemFree(gm)); CHECK(cuMemFree(gv)); CHECK(cuMemFree(gr));
        free(mat); free(x);
    }
    free(out);
}

int main(int argc, char **argv)
{
    Options o = parse_options(argc, argv);
    Dataset data = dataset_alloc();
    real *w = xalloc(NPARAM, sizeof(real));
    dataset_load(o.data, &data);
    model_init(w, o.seed);

    CUdevice device;
    CUcontext context;
    char name[128];
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&device, 0));
    CHECK(cuDeviceGetName(name, sizeof name, device));
    CHECK(cuCtxCreate(&context, 0, device));
    const double begin = now_seconds();
    CUmodule module = load_kernels("src/cuda/kernels.cu");
    printf("gpu name=\"%s\" kernel_build=%.3f\n", name, now_seconds() - begin);

    int status = 0;
    if (o.bench) {
        benchmark(&data, w);
    } else {
        Gpu d = gpu_create(&data, w, o.batch);
        size_t free_bytes, total_bytes;
        CHECK(cuMemGetInfo(&free_bytes, &total_bytes));
        printf("gpu vram_free=%.2f vram_total=%.2f\n", free_bytes / 1073741824.0, total_bytes / 1073741824.0);
        if (o.check) status = check_gradient(&o, &d, &data, w);
        else train(&o, &d);
    }

    CHECK(cuModuleUnload(module));
    CHECK(cuCtxDestroy(context)); /* also releases the buffers */
    dataset_free(&data);
    free(w);
    return status;
}
