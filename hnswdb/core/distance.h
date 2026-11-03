/**
 * @file distance.h
 * @brief Distance/similarity functions for vector comparison
 *
 * This module provides various distance metrics for comparing vectors.
 *
 * Design Principles:
 * - Open/Closed Principle: New distance functions can be added without
 *   modifying existing code
 * - Strategy Pattern: Distance functions are interchangeable via function
 *   pointers
 * - Single Responsibility: Only handles distance computation
 *
 * Supported Metrics:
 * - Cosine distance (1 - cosine similarity)
 * - Euclidean (L2) distance
 * - Inner product (negative, so lower = more similar)
 *
 * Usage:
 *   HNSWDistanceFunc func = distance_get_function(HNSW_DISTANCE_COSINE);
 *   float dist = func(vec_a, vec_b, dimension);
 */

#ifndef HNSW_DISTANCE_H
#define HNSW_DISTANCE_H

#include <stdint.h>
#include "hnsw_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Distance Functions ========== */

/**
 * @brief Compute cosine distance between two vectors
 *
 * Cosine distance = 1 - cosine_similarity
 * Range: [0, 2] where 0 = identical, 2 = opposite
 *
 * @param a First vector
 * @param b Second vector
 * @param dim Vector dimension
 * @return Cosine distance
 */
float distance_cosine(const float* a, const float* b, int32_t dim);

/**
 * @brief Compute squared Euclidean (L2) distance
 *
 * Returns squared distance for efficiency (avoids sqrt).
 * Range: [0, inf) where 0 = identical
 *
 * @param a First vector
 * @param b Second vector
 * @param dim Vector dimension
 * @return Squared L2 distance
 */
float distance_euclidean_sq(const float* a, const float* b, int32_t dim);

/**
 * @brief Compute Euclidean (L2) distance
 *
 * Range: [0, inf) where 0 = identical
 *
 * @param a First vector
 * @param b Second vector
 * @param dim Vector dimension
 * @return L2 distance
 */
float distance_euclidean(const float* a, const float* b, int32_t dim);

/**
 * @brief Compute negative inner product
 *
 * Returns negative inner product so that lower = more similar.
 * Range: (-inf, inf)
 *
 * @param a First vector
 * @param b Second vector
 * @param dim Vector dimension
 * @return Negative inner product
 */
float distance_inner_product(const float* a, const float* b, int32_t dim);

/* ========== Factory Function ========== */

/**
 * @brief Get distance function for a given distance type
 *
 * Factory Pattern: Returns appropriate function pointer based on type.
 *
 * @param type Distance type enum
 * @return Function pointer for computing distance
 */
HNSWDistanceFunc distance_get_function(HNSWDistanceType type);

/**
 * @brief Get human-readable name for distance type
 *
 * @param type Distance type enum
 * @return String name (do not free)
 */
const char* distance_get_name(HNSWDistanceType type);

/* ========== Utility Functions ========== */

/**
 * @brief Compute L2 norm of a vector
 *
 * @param v Vector
 * @param dim Dimension
 * @return L2 norm (sqrt of sum of squares)
 */
float distance_l2_norm(const float* v, int32_t dim);

/**
 * @brief Normalize a vector in place
 *
 * @param v Vector (modified in place)
 * @param dim Dimension
 */
void distance_normalize(float* v, int32_t dim);

/**
 * @brief Compute dot product of two vectors
 *
 * @param a First vector
 * @param b Second vector
 * @param dim Dimension
 * @return Dot product
 */
float distance_dot_product(const float* a, const float* b, int32_t dim);

#ifdef __cplusplus
}
#endif

#endif /* HNSW_DISTANCE_H */
