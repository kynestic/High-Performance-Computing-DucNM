/*
 * CNN: input 1x28x28
 *   -> conv 8@5x5  -> ReLU -> max-pool 2x2   (8x14x14)
 *   -> conv 16@5x5 -> ReLU -> max-pool 2x2   (16x7x7)
 *   -> dense 64    -> ReLU -> dense 10 -> softmax cross-entropy
 *
 * All parameters live in one flat array so that MPI and CUDA can move them at once.
 */
#ifndef MODEL_H
#define MODEL_H

#ifdef USE_DOUBLE
typedef double real; /* only for the numerical gradient check */
#else
typedef float real;
#endif

enum {
    IMG = 28, PIXELS = IMG * IMG,
    KS = 5, PAD = KS / 2,
    C1 = 8, S1 = IMG / 2,  /* channels and side length after block 1 */
    C2 = 16, S2 = S1 / 2,  /* channels and side length after block 2 */
    FLAT = C2 * S2 * S2, HIDDEN = 64, CLASSES = 10,

    /* each layer stores its weights, immediately followed by its biases */
    P_CONV1 = 0,
    P_CONV2 = P_CONV1 + C1 * KS * KS + C1,
    P_DENSE1 = P_CONV2 + C2 * C1 * KS * KS + C2,
    P_DENSE2 = P_DENSE1 + HIDDEN * FLAT + HIDDEN,
    NPARAM = P_DENSE2 + CLASSES * HIDDEN + CLASSES
};

void model_init(real *w, int seed);
/* Adds the gradient of one sample to g and returns its loss. */
real model_gradient(const real *w, const real *image, int label, real *g);
real model_loss(const real *w, const real *image, int label);
int model_predict(const real *w, const real *image);
void sgd_update(real *w, real *velocity, const real *g, real scale, real lr, real momentum);

/* Single layers, also used by the CPU side of the layer benchmark. */
void conv_forward(const real *in, real *out, const real *w, int cin, int cout, int side);
void dense_forward(const real *in, real *out, const real *w, int nin, int nout, int relu);

#endif
