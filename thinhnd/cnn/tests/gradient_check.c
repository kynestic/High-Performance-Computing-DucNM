/* Backpropagation against central finite differences, compiled in double precision. */
#include "../src/common/train_common.h"

int main(void)
{
    Dataset d = dataset_alloc();
    real *w = xalloc(NPARAM, sizeof(real)), *g = xalloc(NPARAM, sizeof(real));
    const int start[5] = {P_CONV1, P_CONV2, P_DENSE1, P_DENSE2, NPARAM};
    const double h = 1e-6;
    uint64_t state = 7;
    int failed = 0;

    dataset_load("data", &d);
    model_init(w, 5408);
    for (int s = 0; s < 4; s++) model_gradient(w, d.x + s * PIXELS, d.y[s], g);

    for (int layer = 0; layer < 4; layer++) {
        double worst = 0;
        for (int t = 0; t < 60; t++) {
            /* the last 5 probes hit the biases */
            const int size = start[layer + 1] - start[layer];
            const int i = start[layer] + (t < 55 ? (int)(rng_next(&state) % size) : size - 1 - (t - 55));
            const real old = w[i];
            double plus = 0, minus = 0;
            w[i] = old + h;
            for (int s = 0; s < 4; s++) plus += model_loss(w, d.x + s * PIXELS, d.y[s]);
            w[i] = old - h;
            for (int s = 0; s < 4; s++) minus += model_loss(w, d.x + s * PIXELS, d.y[s]);
            w[i] = old;
            const double numeric = (plus - minus) / (2 * h);
            const double error = fabs(numeric - g[i]) / fmax(1e-3, fabs(numeric) + fabs(g[i]));
            worst = fmax(worst, error);
        }
        printf("layer=%d worst_relative_error=%.3e %s\n", layer + 1, worst, worst < 1e-4 ? "PASS" : "FAIL");
        failed |= worst >= 1e-4;
    }
    dataset_free(&d);
    free(w); free(g);
    return failed;
}
