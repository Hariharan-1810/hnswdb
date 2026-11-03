/**
 * @file distance.c
 * @brief Distance function implementations
 *
 * Implementation Notes:
 * - SIMD optimizations can be added here without affecting other code
 * - All functions handle edge cases (zero vectors, etc.)
 * - Numerical stability is prioritized
 */

#include "distance.h"
#include <math.h>

/* ========== Constants ========== */

/** Epsilon for floating point comparisons */
#define EPSILON 1e-10f

/* ========== Core Distance Functions ========== */

float distance_cosine(const float* a, const float* b, int32_t dim) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    /* Compute dot product and norms in single pass */
    for (int32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    /* Handle zero vectors */
    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < EPSILON) {
        return 1.0f;  /* Maximum distance for undefined case */
    }

    float similarity = dot / denom;

    /* Clamp to [-1, 1] to handle floating point errors */
    if (similarity > 1.0f) similarity = 1.0f;
    if (similarity < -1.0f) similarity = -1.0f;

    return 1.0f - similarity;
}

float distance_euclidean_sq(const float* a, const float* b, int32_t dim) {
    float sum = 0.0f;

    for (int32_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

float distance_euclidean(const float* a, const float* b, int32_t dim) {
    return sqrtf(distance_euclidean_sq(a, b, dim));
}

float distance_inner_product(const float* a, const float* b, int32_t dim) {
    float dot = 0.0f;

    for (int32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }

    /* Return negative so lower = more similar */
    return -dot;
}

/* ========== Factory Function ========== */

HNSWDistanceFunc distance_get_function(HNSWDistanceType type) {
    switch (type) {
        case HNSW_DISTANCE_COSINE:
            return distance_cosine;
        case HNSW_DISTANCE_EUCLIDEAN:
            return distance_euclidean;
        case HNSW_DISTANCE_INNER_PRODUCT:
            return distance_inner_product;
        default:
            return distance_cosine;  /* Default fallback */
    }
}

const char* distance_get_name(HNSWDistanceType type) {
    switch (type) {
        case HNSW_DISTANCE_COSINE:
            return "cosine";
        case HNSW_DISTANCE_EUCLIDEAN:
            return "euclidean";
        case HNSW_DISTANCE_INNER_PRODUCT:
            return "inner_product";
        default:
            return "unknown";
    }
}

/* ========== Utility Functions ========== */

float distance_l2_norm(const float* v, int32_t dim) {
    float sum = 0.0f;

    for (int32_t i = 0; i < dim; i++) {
        sum += v[i] * v[i];
    }

    return sqrtf(sum);
}

void distance_normalize(float* v, int32_t dim) {
    float norm = distance_l2_norm(v, dim);

    if (norm < EPSILON) {
        return;  /* Cannot normalize zero vector */
    }

    float inv_norm = 1.0f / norm;
    for (int32_t i = 0; i < dim; i++) {
        v[i] *= inv_norm;
    }
}

float distance_dot_product(const float* a, const float* b, int32_t dim) {
    float dot = 0.0f;

    for (int32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }

    return dot;
}
