/*
 * All GPU kernels (no cuBLAS / cuDNN). One thread computes one output element, so no two
 * threads write to the same address and no atomics are needed. Compiled at run time by NVRTC.
 */
#define THREAD_ID ((int)(blockIdx.x * blockDim.x + threadIdx.x))

static __device__ inline int max_i(int a, int b) { return a > b ? a : b; }
static __device__ inline int min_i(int a, int b) { return a < b ? a : b; }

/* Gather the shuffled mini-batch. n = m * pixels */
extern "C" __global__ void load_batch(const float *data, const int *perm, float *out, int m, int pixels)
{
    const int id = THREAD_ID;
    if (id >= m * pixels) return;
    out[id] = data[perm[id / pixels] * pixels + id % pixels];
}

/* n = m * cout * side * side */
extern "C" __global__ void conv_forward(const float *in, float *out, const float *w,
                                        int m, int cin, int cout, int side, int ks)
{
    const int id = THREAD_ID, area = side * side;
    if (id >= m * cout * area) return;
    const int pos = id % area, o = (id / area) % cout, s = id / (area * cout);
    const int y = pos / side, x = pos % side, pad = ks / 2;
    const int a0 = max_i(0, pad - y), a1 = min_i(ks, side + pad - y);
    const int b0 = max_i(0, pad - x), b1 = min_i(ks, side + pad - x);

    float sum = w[cout * cin * ks * ks + o];
    for (int c = 0; c < cin; c++) {
        const float *src = in + (s * cin + c) * area;
        const float *k = w + (o * cin + c) * ks * ks;
        for (int a = a0; a < a1; a++)
            for (int b = b0; b < b1; b++)
                sum += k[a * ks + b] * src[(y + a - pad) * side + x + b - pad];
    }
    out[id] = sum > 0.f ? sum : 0.f;
}

/* n = m * channels * half * half */
extern "C" __global__ void pool_forward(const float *in, float *out, int *at, int m, int channels, int side)
{
    const int id = THREAD_ID, half = side / 2, area = half * half;
    if (id >= m * channels * area) return;
    const int pos = id % area, plane = id / area;
    const int first = (plane * side + 2 * (pos / half)) * side + 2 * (pos % half);

    int best = first;
    if (in[first + 1] > in[best]) best = first + 1;
    if (in[first + side] > in[best]) best = first + side;
    if (in[first + side + 1] > in[best]) best = first + side + 1;
    out[id] = in[best];
    at[id] = best;
}

/* n = m * channels * side * side */
extern "C" __global__ void pool_relu_backward(const float *dout, const int *at, const float *act, float *din,
                                              int m, int channels, int side)
{
    const int id = THREAD_ID, half = side / 2, area = side * side;
    if (id >= m * channels * area) return;
    const int pos = id % area, plane = id / area;
    const int pooled = (plane * half + pos / side / 2) * half + (pos % side) / 2;
    din[id] = (at[pooled] == id && act[id] > 0.f) ? dout[pooled] : 0.f;
}

/* n = m * cin * side * side */
extern "C" __global__ void conv_backward_input(const float *dout, float *din, const float *w,
                                               int m, int cin, int cout, int side, int ks)
{
    const int id = THREAD_ID, area = side * side;
    if (id >= m * cin * area) return;
    const int pos = id % area, c = (id / area) % cin, s = id / (area * cin);
    const int y = pos / side, x = pos % side, pad = ks / 2;
    const int a0 = max_i(0, y + pad - side + 1), a1 = min_i(ks, y + pad + 1);
    const int b0 = max_i(0, x + pad - side + 1), b1 = min_i(ks, x + pad + 1);

    float sum = 0.f;
    for (int o = 0; o < cout; o++) {
        const float *src = dout + (s * cout + o) * area;
        const float *k = w + (o * cin + c) * ks * ks;
        for (int a = a0; a < a1; a++)
            for (int b = b0; b < b1; b++)
                sum += k[a * ks + b] * src[(y - a + pad) * side + x - b + pad];
    }
    din[id] = sum;
}

/* Map step: gradient of every parameter for every sample. n = m * (weights + cout) */
extern "C" __global__ void conv_backward_weight(const float *in, const float *dout, float *part,
                                                int m, int cin, int cout, int side, int ks)
{
    const int id = THREAD_ID, weights = cout * cin * ks * ks, count = weights + cout, area = side * side;
    if (id >= m * count) return;
    const int s = id / count, p = id % count;
    float sum = 0.f;

    if (p >= weights) {
        const float *src = dout + (s * cout + p - weights) * area;
        for (int i = 0; i < area; i++) sum += src[i];
    } else {
        const int b = p % ks, a = (p / ks) % ks, plane = p / (ks * ks), pad = ks / 2;
        const float *src = dout + (s * cout + plane / cin) * area;
        const float *input = in + (s * cin + plane % cin) * area;
        const int y0 = max_i(0, pad - a), y1 = min_i(side, side + pad - a);
        const int x0 = max_i(0, pad - b), x1 = min_i(side, side + pad - b);
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++)
                sum += src[y * side + x] * input[(y + a - pad) * side + x + b - pad];
    }
    part[id] = sum;
}

/* Reduce step: sum over the samples. n = count */
extern "C" __global__ void sum_samples(const float *part, float *g, int m, int count)
{
    const int p = THREAD_ID;
    if (p >= count) return;
    float sum = 0.f;
    for (int s = 0; s < m; s++) sum += part[s * count + p];
    g[p] = sum;
}

/* n = m * nout */
extern "C" __global__ void dense_forward(const float *in, float *out, const float *w,
                                         int m, int nin, int nout, int relu)
{
    const int id = THREAD_ID;
    if (id >= m * nout) return;
    const int o = id % nout;
    const float *x = in + (id / nout) * nin, *row = w + o * nin;
    float sum = w[nout * nin + o];
    for (int i = 0; i < nin; i++) sum += row[i] * x[i];
    out[id] = (relu && sum < 0.f) ? 0.f : sum;
}

/* act != 0: the output of this layer went through a ReLU. n = m * nin */
extern "C" __global__ void dense_backward_input(const float *dout, const float *act, float *din, const float *w,
                                                int m, int nin, int nout)
{
    const int id = THREAD_ID;
    if (id >= m * nin) return;
    const int base = (id / nin) * nout, i = id % nin;
    float sum = 0.f;
    for (int o = 0; o < nout; o++)
        if (!act || act[base + o] > 0.f) sum += w[o * nin + i] * dout[base + o];
    din[id] = sum;
}

/* n = nout * nin + nout */
extern "C" __global__ void dense_backward_weight(const float *in, const float *dout, const float *act, float *g,
                                                 int m, int nin, int nout)
{
    const int p = THREAD_ID, weights = nout * nin;
    if (p >= weights + nout) return;
    const int bias = p >= weights;
    const int o = bias ? p - weights : p / nin, i = bias ? 0 : p % nin;
    float sum = 0.f;
    for (int s = 0; s < m; s++)
        if (!act || act[s * nout + o] > 0.f) sum += dout[s * nout + o] * (bias ? 1.f : in[s * nin + i]);
    g[p] = sum;
}

/* n = m */
extern "C" __global__ void softmax(const float *logit, const int *label, const int *perm,
                                   float *df, float *loss, int *correct, int m, int classes)
{
    const int s = THREAD_ID;
    if (s >= m) return;
    const float *f = logit + s * classes;
    const int y = label[perm[s]];
    int best = 0;
    for (int j = 1; j < classes; j++)
        if (f[j] > f[best]) best = j;

    float sum = 0.f;
    for (int j = 0; j < classes; j++) {
        const float e = expf(f[j] - f[best]);
        df[s * classes + j] = e;
        sum += e;
    }
    for (int j = 0; j < classes; j++) df[s * classes + j] = df[s * classes + j] / sum - (j == y ? 1.f : 0.f);
    loss[s] = f[best] - f[y] + logf(sum);
    correct[s] = best == y;
}

/* n = count */
extern "C" __global__ void sgd_update(float *w, float *v, const float *g, int count,
                                      float scale, float lr, float momentum)
{
    const int p = THREAD_ID;
    if (p >= count) return;
    v[p] = momentum * v[p] + scale * g[p];
    w[p] -= lr * v[p];
}
