/**
 * @file hnsw.h
 * @brief HNSW Vector Database - Public API
 *
 * High-performance approximate nearest neighbor search using the
 * Hierarchical Navigable Small World (HNSW) algorithm.
 *
 * Features:
 * - O(log n) search complexity
 * - Cosine, Euclidean, and Inner Product distance metrics
 * - Automatic dimension detection
 * - Persistent storage to disk
 *
 * Basic Usage:
 * @code
 *   HNSWIndex* index = hnsw_create(0);  // 0 = auto-detect dimension
 *   hnsw_add(index, vectors, count, dim);
 *   hnsw_search(index, query, dim, k, results, &num_results);
 *   hnsw_save(index, "index.hnsw");
 *   hnsw_destroy(index);
 * @endcode
 *
 * @author Hariharan Thangaraj
 * @version 0.1.0
 *
 * @see https://arxiv.org/abs/1603.09320 - Original HNSW paper
 */

#ifndef HNSW_H
#define HNSW_H

#include <stdint.h>
#include <stddef.h>

/* Include platform-specific definitions */
#include "platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Forward Declarations ========== */

/**
 * @struct HNSWIndex
 * @brief Opaque handle to an HNSW index
 *
 * The internal structure is hidden to allow implementation changes
 * without breaking the API.
 */
typedef struct HNSWIndex HNSWIndex;

/* ========== Error Codes ========== */

/** Operation successful */
#define HNSW_OK                     0
/** Memory allocation failed */
#define HNSW_ERR_MEMORY            -1
/** Invalid parameter provided */
#define HNSW_ERR_INVALID_PARAM     -2
/** Index not initialized */
#define HNSW_ERR_NOT_INITIALIZED   -3
/** File I/O error */
#define HNSW_ERR_FILE_IO           -4
/** Vector dimension mismatch */
#define HNSW_ERR_DIMENSION_MISMATCH -5
/** Index is empty */
#define HNSW_ERR_EMPTY_INDEX       -6
/** Invalid file format */
#define HNSW_ERR_INVALID_FILE      -7
/** File version not supported */
#define HNSW_ERR_VERSION_MISMATCH  -8

/* ========== Result Structure ========== */

/**
 * @struct HNSWResult
 * @brief Result from k-NN search
 */
typedef struct {
    int32_t id;         /**< Vector ID (0-indexed) */
    float   distance;   /**< Distance to query (lower = more similar) */
} HNSWResult;

/* ========== Index Lifecycle ========== */

/**
 * @brief Create a new HNSW index
 *
 * Creates an empty index ready for storing vectors.
 *
 * @param dim Vector dimension. Use 0 for auto-detection from first insert.
 * @return Pointer to new index, or NULL on allocation failure
 *
 * @note Caller is responsible for calling hnsw_destroy()
 *
 * Example:
 * @code
 *   HNSWIndex* index = hnsw_create(384);  // 384-dimensional vectors
 *   if (!index) {
 *       // Handle error
 *   }
 * @endcode
 */
HNSW_API HNSWIndex* hnsw_create(int32_t dim);

/**
 * @brief Destroy an HNSW index and free all memory
 *
 * @param index Index to destroy (NULL-safe)
 *
 * @note All resources are freed; the pointer becomes invalid
 */
HNSW_API void hnsw_destroy(HNSWIndex* index);

/* ========== Vector Operations ========== */

/**
 * @brief Add vectors to the index
 *
 * Inserts one or more vectors into the HNSW graph. The first insertion
 * sets the dimension if auto-detection was enabled.
 *
 * @param index   Target index
 * @param vectors Flat array of vectors (n * dim floats, row-major)
 * @param n       Number of vectors to add
 * @param dim     Dimension of each vector
 * @return HNSW_OK on success, error code otherwise
 *
 * Error Codes:
 * - HNSW_ERR_INVALID_PARAM: NULL pointer or invalid dimensions
 * - HNSW_ERR_DIMENSION_MISMATCH: dim doesn't match existing vectors
 * - HNSW_ERR_MEMORY: Allocation failure
 *
 * Example:
 * @code
 *   float vectors[] = {
 *       0.1f, 0.2f, 0.3f,  // Vector 0
 *       0.4f, 0.5f, 0.6f,  // Vector 1
 *   };
 *   int result = hnsw_add(index, vectors, 2, 3);
 * @endcode
 */
HNSW_API int hnsw_add(HNSWIndex* index, const float* vectors, int32_t n, int32_t dim);

/**
 * @brief Search for k nearest neighbors
 *
 * Finds the k vectors closest to the query using the HNSW graph.
 *
 * @param index       Index to search
 * @param query       Query vector (dim floats)
 * @param dim         Query dimension (must match index)
 * @param k           Number of neighbors to find
 * @param results     Output array (must be pre-allocated for k results)
 * @param num_results Output: actual number of results found (may be < k)
 * @return HNSW_OK on success, error code otherwise
 *
 * @note Results are sorted by distance (closest first)
 *
 * Example:
 * @code
 *   HNSWResult results[10];
 *   int32_t found;
 *   float query[] = {0.1f, 0.2f, 0.3f};
 *
 *   int status = hnsw_search(index, query, 3, 10, results, &found);
 *   for (int i = 0; i < found; i++) {
 *       printf("ID: %d, Distance: %f\n", results[i].id, results[i].distance);
 *   }
 * @endcode
 */
HNSW_API int hnsw_search(HNSWIndex* index, const float* query, int32_t dim,
                          int32_t k, HNSWResult* results, int32_t* num_results);

/**
 * @brief Get all vectors from the index
 *
 * Retrieves all stored vectors. Call with vectors=NULL to get counts first.
 *
 * @param index   Index to query
 * @param vectors Output array (n * dim floats) or NULL to get counts only
 * @param count   Output: number of vectors
 * @param dim     Output: vector dimension
 * @return HNSW_OK on success, error code otherwise
 *
 * Example:
 * @code
 *   int32_t count, dim;
 *   hnsw_get_vectors(index, NULL, &count, &dim);
 *
 *   float* vectors = malloc(count * dim * sizeof(float));
 *   hnsw_get_vectors(index, vectors, &count, &dim);
 * @endcode
 */
HNSW_API int hnsw_get_vectors(HNSWIndex* index, float* vectors,
                               int32_t* count, int32_t* dim);

/* ========== Persistence ========== */

/**
 * @brief Save index to file
 *
 * Persists the complete index to disk in binary format.
 *
 * @param index    Index to save
 * @param filepath Output file path
 * @return HNSW_OK on success, error code otherwise
 *
 * @note Overwrites existing file without warning
 */
HNSW_API int hnsw_save(HNSWIndex* index, const char* filepath);

/**
 * @brief Load index from file
 *
 * Loads a previously saved index from disk.
 *
 * @param filepath Path to saved file
 * @return Pointer to loaded index, or NULL on failure
 *
 * @note Caller is responsible for calling hnsw_destroy()
 */
HNSW_API HNSWIndex* hnsw_load(const char* filepath);

/* ========== Index Properties ========== */

/**
 * @brief Get number of vectors in index
 * @param index Index to query
 * @return Number of vectors, or -1 on error
 */
HNSW_API int32_t hnsw_count(HNSWIndex* index);

/**
 * @brief Get vector dimension
 * @param index Index to query
 * @return Dimension, or -1 if not set
 */
HNSW_API int32_t hnsw_dimension(HNSWIndex* index);

/**
 * @brief Check if index has been modified since last save
 * @param index Index to query
 * @return 1 if modified, 0 if not, -1 on error
 */
HNSW_API int hnsw_is_modified(HNSWIndex* index);

/**
 * @brief Mark index as saved (clears modified flag)
 * @param index Index to update
 */
HNSW_API void hnsw_mark_saved(HNSWIndex* index);

/* ========== Error Handling ========== */

/**
 * @brief Get human-readable error message
 * @param error_code Error code returned by HNSW function
 * @return Error message string (do not free)
 */
HNSW_API const char* hnsw_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif /* HNSW_H */
