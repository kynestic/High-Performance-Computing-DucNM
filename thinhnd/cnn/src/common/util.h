/* Utilities shared by all programs: error handling, clock, random numbers. */
#ifndef UTIL_H
#define UTIL_H

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00 /* Windows 10, for SetProcessInformation */
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

/* Windows slows down (or moves to E-cores) processes without a window; opt out. */
static inline void process_setup(void)
{
#ifdef _WIN32
    PROCESS_POWER_THROTTLING_STATE s = {PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                                        PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0};
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &s, sizeof s);
#endif
}

static inline void die(const char *what, const char *detail)
{
    fprintf(stderr, "error: %s %s\n", what, detail ? detail : "");
    exit(EXIT_FAILURE);
}

static inline void *xalloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size);
    if (!p) die("out of memory", NULL);
    return p;
}

static inline double now_seconds(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* SplitMix64: gives the same sequence in every implementation. */
static inline uint64_t rng_next(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline double rng_uniform(uint64_t *state)
{
    return ((rng_next(state) >> 11) + 0.5) / 9007199254740992.0;
}

static inline double rng_normal(uint64_t *state)
{
    double u = rng_uniform(state), v = rng_uniform(state);
    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

/* Fisher-Yates shuffle determined only by (seed, epoch). */
static inline void make_permutation(int *perm, int n, int seed, int epoch)
{
    uint64_t state = (uint64_t)seed * 1000003ULL + (uint64_t)epoch;
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rng_next(&state) % (uint64_t)(i + 1));
        int t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }
}

/* Start of part r when [0, n) is divided into p contiguous parts. */
static inline int part_begin(int n, int p, int r)
{
    return n / p * r + (n % p < r ? n % p : r);
}

#endif
