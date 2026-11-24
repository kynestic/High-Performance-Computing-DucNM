/* Forward and backward propagation, written without any numerical library. */
#include "model.h"
#include "util.h"

typedef struct {
    real a1[C1 * IMG * IMG], p1[C1 * S1 * S1];
    real a2[C2 * S1 * S1], p2[FLAT];
    real h[HIDDEN], logit[CLASSES];
    int at1[C1 * S1 * S1], at2[FLAT]; /* position of each pooled maximum */
} Activations;

static int max_i(int a, int b) { return a > b ? a : b; }
static int min_i(int a, int b) { return a < b ? a : b; }

static void init_layer(real *w, int count, int biases, double std, uint64_t *state)
{
    for (int i = 0; i < count; i++) w[i] = (real)(std * rng_normal(state));
    for (int i = 0; i < biases; i++) w[count + i] = 0;
}

void model_init(real *w, int seed)
{
    uint64_t state = (uint64_t)seed;
    init_layer(w + P_CONV1, C1 * KS * KS, C1, sqrt(2.0 / (KS * KS)), &state);
    init_layer(w + P_CONV2, C2 * C1 * KS * KS, C2, sqrt(2.0 / (C1 * KS * KS)), &state);
    init_layer(w + P_DENSE1, HIDDEN * FLAT, HIDDEN, sqrt(2.0 / FLAT), &state);
    init_layer(w + P_DENSE2, CLASSES * HIDDEN, CLASSES, sqrt(1.0 / HIDDEN), &state);
}

/* out[o][y][x] = ReLU(b[o] + sum w[o][c][a][b] * in[c][y+a-PAD][x+b-PAD]) */
void conv_forward(const real *in, real *out, const real *w, int cin, int cout, int side)
{
    const int area = side * side;
    const real *bias = w + cout * cin * KS * KS;

    for (int o = 0; o < cout; o++) {
        real *dst = out + o * area;
        for (int i = 0; i < area; i++) dst[i] = bias[o];
        for (int c = 0; c < cin; c++)
            for (int a = 0; a < KS; a++)
                for (int b = 0; b < KS; b++) {
                    const real value = w[((o * cin + c) * KS + a) * KS + b];
                    const int y0 = max_i(0, PAD - a), y1 = min_i(side, side + PAD - a);
                    const int x0 = max_i(0, PAD - b), x1 = min_i(side, side + PAD - b);
                    const int shift = c * area + (a - PAD) * side + (b - PAD);
                    for (int y = y0; y < y1; y++)
                        for (int x = x0; x < x1; x++)
                            dst[y * side + x] += value * in[y * side + x + shift];
                }
        for (int i = 0; i < area; i++) dst[i] = dst[i] > 0 ? dst[i] : 0;
    }
}

/* dout must already be masked by ReLU. din may be NULL for the first layer. */
static void conv_backward(const real *in, const real *dout, real *din, const real *w, real *g,
                          int cin, int cout, int side)
{
    const int area = side * side, weights = cout * cin * KS * KS;

    if (din) memset(din, 0, sizeof(real) * cin * area);
    for (int o = 0; o < cout; o++) {
        const real *src = dout + o * area;
        real sum = 0;
        for (int i = 0; i < area; i++) sum += src[i];
        g[weights + o] += sum;

        for (int c = 0; c < cin; c++)
            for (int a = 0; a < KS; a++)
                for (int b = 0; b < KS; b++) {
                    const int k = ((o * cin + c) * KS + a) * KS + b;
                    const int y0 = max_i(0, PAD - a), y1 = min_i(side, side + PAD - a);
                    const int x0 = max_i(0, PAD - b), x1 = min_i(side, side + PAD - b);
                    const int shift = c * area + (a - PAD) * side + (b - PAD);
                    const real value = w[k];
                    real dot = 0;
                    for (int y = y0; y < y1; y++) {
                        for (int x = x0; x < x1; x++)
                            dot += src[y * side + x] * in[y * side + x + shift];
                        if (din)
                            for (int x = x0; x < x1; x++)
                                din[y * side + x + shift] += value * src[y * side + x];
                    }
                    g[k] += dot;
                }
    }
}

static void pool_forward(const real *in, real *out, int *at, int channels, int side)
{
    const int half = side / 2;
    for (int c = 0; c < channels; c++)
        for (int y = 0; y < half; y++)
            for (int x = 0; x < half; x++) {
                const int first = (c * side + 2 * y) * side + 2 * x;
                const int cell[4] = {first, first + 1, first + side, first + side + 1};
                int best = first;
                for (int i = 1; i < 4; i++)
                    if (in[cell[i]] > in[best]) best = cell[i];
                const int o = (c * half + y) * half + x;
                out[o] = in[best];
                at[o] = best;
            }
}

/* Pooling passes the gradient only to the maximum; ReLU only where the output is positive. */
static void pool_relu_backward(const real *dout, const int *at, const real *act, real *din,
                               int channels, int side)
{
    const int half = side / 2;
    memset(din, 0, sizeof(real) * channels * side * side);
    for (int i = 0; i < channels * half * half; i++)
        if (act[at[i]] > 0) din[at[i]] = dout[i];
}

void dense_forward(const real *in, real *out, const real *w, int nin, int nout, int relu)
{
    for (int o = 0; o < nout; o++) {
        const real *row = w + o * nin;
        real sum = w[nout * nin + o];
        for (int i = 0; i < nin; i++) sum += row[i] * in[i];
        out[o] = relu && sum < 0 ? 0 : sum;
    }
}

static void dense_backward(const real *in, const real *dout, real *din, const real *w, real *g,
                           int nin, int nout)
{
    memset(din, 0, sizeof(real) * nin);
    for (int o = 0; o < nout; o++) {
        const real d = dout[o];
        if (d == 0) continue;
        g[nout * nin + o] += d;
        for (int i = 0; i < nin; i++) g[o * nin + i] += d * in[i];
        for (int i = 0; i < nin; i++) din[i] += d * w[o * nin + i];
    }
}

static void forward(const real *w, const real *image, Activations *a)
{
    conv_forward(image, a->a1, w + P_CONV1, 1, C1, IMG);
    pool_forward(a->a1, a->p1, a->at1, C1, IMG);
    conv_forward(a->p1, a->a2, w + P_CONV2, C1, C2, S1);
    pool_forward(a->a2, a->p2, a->at2, C2, S1);
    dense_forward(a->p2, a->h, w + P_DENSE1, FLAT, HIDDEN, 1);
    dense_forward(a->h, a->logit, w + P_DENSE2, HIDDEN, CLASSES, 0);
}

/* L = -f_y + log(sum exp f_j) and dL/df_j = softmax_j - [j == y] */
static real softmax_loss(const real *f, int label, real *df)
{
    real top = f[0], sum = 0, e[CLASSES];
    for (int j = 1; j < CLASSES; j++) top = f[j] > top ? f[j] : top;
    for (int j = 0; j < CLASSES; j++) sum += e[j] = (real)exp(f[j] - top); /* avoids overflow */
    if (df)
        for (int j = 0; j < CLASSES; j++) df[j] = e[j] / sum - (j == label);
    return top - f[label] + (real)log(sum);
}

real model_gradient(const real *w, const real *image, int label, real *g)
{
    Activations a;
    real df[CLASSES], dh[HIDDEN], dp2[FLAT], da2[C2 * S1 * S1], dp1[C1 * S1 * S1], da1[C1 * IMG * IMG];

    forward(w, image, &a);
    const real loss = softmax_loss(a.logit, label, df);

    dense_backward(a.h, df, dh, w + P_DENSE2, g + P_DENSE2, HIDDEN, CLASSES);
    for (int i = 0; i < HIDDEN; i++)
        if (!(a.h[i] > 0)) dh[i] = 0;
    dense_backward(a.p2, dh, dp2, w + P_DENSE1, g + P_DENSE1, FLAT, HIDDEN);
    pool_relu_backward(dp2, a.at2, a.a2, da2, C2, S1);
    conv_backward(a.p1, da2, dp1, w + P_CONV2, g + P_CONV2, C1, C2, S1);
    pool_relu_backward(dp1, a.at1, a.a1, da1, C1, IMG);
    conv_backward(image, da1, NULL, w + P_CONV1, g + P_CONV1, 1, C1, IMG);
    return loss;
}

real model_loss(const real *w, const real *image, int label)
{
    Activations a;
    forward(w, image, &a);
    return softmax_loss(a.logit, label, NULL);
}

int model_predict(const real *w, const real *image)
{
    Activations a;
    int best = 0;
    forward(w, image, &a);
    for (int j = 1; j < CLASSES; j++)
        if (a.logit[j] > a.logit[best]) best = j;
    return best;
}

void sgd_update(real *w, real *velocity, const real *g, real scale, real lr, real momentum)
{
    for (int i = 0; i < NPARAM; i++) {
        velocity[i] = momentum * velocity[i] + scale * g[i];
        w[i] -= lr * velocity[i];
    }
}
