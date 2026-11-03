/**
 * @file hnsw_types.h
 * @brief Internal type definitions for HNSW implementation
 *
 * This header defines all internal data structures used by the HNSW
 * algorithm. These types are not exposed in the public API.
 *
 * Design Principles:
 * - Encapsulation: Internal structures hidden from users
 * - Single Responsibility: Each struct has a clear purpose
 * - Data Locality: Related data grouped together
 */

#ifndef HNSW_TYPES_H
#define HNSW_TYPES_H

#include <stdint.h>
#include "platform.h"
#include "hnsw.h"   /* public API: error codes, HNSWResult, HNSWIndex typedef */

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Configuration Constants ========== */

/**
 * HNSW Algorithm Parameters
 *
 * M: Maximum number of connections per node per layer
 *    - Higher M = better recall, more memory, slower build
 *    - We use M=18 for good balance of speed and accuracy
 *
 * M_MAX: Maximum connections at layer 0 (typically 2*M)
 *        Layer 0 is denser for better search quality
 *
 * Convergent Search: No ef parameter - search continues until
 *                    no candidate can improve the current top-k
 */
#define HNSW_M                 18
#define HNSW_M_MAX             (HNSW_M * 2)

/** Level multiplier for probabilistic level generation */
#define HNSW_LEVEL_MULTIPLIER  (1.0f / 2.885390f)  /* 1/ln(M) where M=18 */

/** Initial node capacity for dynamic array */
#define HNSW_INITIAL_CAPACITY  1024

/** Memory alignment for SIMD operations */
#define HNSW_ALIGNMENT         32

/* Error codes (HNSW_OK, HNSW_ERR_*) are defined once, in the public hnsw.h. */

/* ========== Distance Function Types ========== */

/**
 * @enum HNSWDistanceType
 * @brief Supported distance/similarity metrics
 *
 * Open/Closed Principle: New distance types can be added
 * without modifying existing code.
 */
typedef enum {
    HNSW_DISTANCE_COSINE       = 0,   /**< Cosine distance (1 - similarity) */
    HNSW_DISTANCE_EUCLIDEAN    = 1,   /**< L2 Euclidean distance */
    HNSW_DISTANCE_INNER_PRODUCT = 2   /**< Negative inner product */
} HNSWDistanceType;

/**
 * @typedef HNSWDistanceFunc
 * @brief Function pointer type for distance computation
 *
 * Strategy Pattern: Allows swapping distance functions at runtime.
 *
 * @param a First vector
 * @param b Second vector
 * @param dim Vector dimension
 * @return Distance value (lower = more similar)
 */
typedef float (*HNSWDistanceFunc)(const float* a, const float* b, int32_t dim);

/* ========== Node Structure ========== */

/**
 * @struct HNSWNode
 * @brief Represents a single node in the HNSW graph
 *
 * Each node contains:
 * - The vector data
 * - Connections to neighbors at each level
 * - Metadata (ID, max level)
 *
 * Memory Layout:
 * - neighbors: Flat array [level0_neighbors][level1_neighbors]...
 * - Each level has M_MAX slots for level 0, M slots for higher levels
 */
typedef struct {
    float*    vector;           /**< Vector data (dim floats) */
    int32_t*  neighbors;        /**< Neighbor IDs, flat array for all levels */
    int32_t*  neighbor_counts;  /**< Number of neighbors per level */
    int32_t   max_level;        /**< Maximum level this node exists at */
    int32_t   id;               /**< Unique node identifier */
} HNSWNode;

/* ========== Index Structure ========== */

/**
 * @struct HNSWIndex
 * @brief Main HNSW index structure
 *
 * This is the primary data structure managing the HNSW graph.
 * It owns all nodes and their associated memory.
 *
 * Invariants:
 * - entry_point is valid if node_count > 0
 * - max_level >= 0 if node_count > 0
 * - All nodes with id < node_count are valid
 */
struct HNSWIndex {
    /* Node storage */
    HNSWNode*     nodes;            /**< Array of all nodes */
    int32_t       node_count;       /**< Number of nodes in index */
    int32_t       node_capacity;    /**< Allocated capacity */

    /* Index properties */
    int32_t       dim;              /**< Vector dimension (0 = auto-detect) */
    int32_t       entry_point;      /**< ID of entry point node (-1 if empty) */
    int32_t       max_level;        /**< Current maximum level (-1 if empty) */

    /* Distance computation */
    HNSWDistanceType  distance_type;    /**< Type of distance metric */
    HNSWDistanceFunc  distance_func;    /**< Distance function pointer */

    /* State tracking */
    int           is_modified;      /**< 1 if modified since last save */
    unsigned int  random_seed;      /**< Seed for level generation */
};

/* HNSWResult (search result) is defined in the public hnsw.h. */

/* ========== Utility Macros ========== */

/** Get maximum neighbors allowed at a level */
#define HNSW_MAX_NEIGHBORS(level) ((level) == 0 ? HNSW_M_MAX : HNSW_M)

/** Calculate offset into neighbors array for a level */
#define HNSW_NEIGHBOR_OFFSET(level) ((level) * HNSW_M_MAX)

/** Minimum of two values */
#define HNSW_MIN(a, b) (((a) < (b)) ? (a) : (b))

/** Maximum of two values */
#define HNSW_MAX(a, b) (((a) > (b)) ? (a) : (b))

#ifdef __cplusplus
}
#endif

#endif /* HNSW_TYPES_H */
