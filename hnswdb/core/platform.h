/**
 * @file platform.h
 * @brief Platform-specific compatibility layer
 *
 * This header provides platform-independent abstractions for OS-specific
 * functionality. Follows the Adapter pattern for cross-platform support.
 */

#ifndef HNSW_PLATFORM_H
#define HNSW_PLATFORM_H

#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Export Macros ========== */

#ifdef _WIN32
    #ifdef HNSW_EXPORTS
        #define HNSW_API __declspec(dllexport)
    #else
        #define HNSW_API __declspec(dllimport)
    #endif
    #define HNSW_INLINE __inline
#else
    #define HNSW_API __attribute__((visibility("default")))
    #define HNSW_INLINE inline
#endif

/* ========== Thread-Safe Random ========== */

/**
 * @brief Thread-safe random number generator
 * @param seed Pointer to seed value (modified in place)
 * @return Random integer
 *
 * Windows doesn't have rand_r, so we provide a compatible implementation.
 */
static HNSW_INLINE int hnsw_rand_r(unsigned int* seed) {
#ifdef _WIN32
    /* Linear congruential generator (same as glibc) */
    *seed = *seed * 1103515245 + 12345;
    return (int)((*seed / 65536) % 32768);
#else
    return rand_r(seed);
#endif
}

/* ========== Memory Alignment ========== */

/**
 * @brief Allocate aligned memory for SIMD operations
 * @param size Number of bytes to allocate
 * @param alignment Alignment boundary (must be power of 2)
 * @return Pointer to aligned memory or NULL on failure
 */
static HNSW_INLINE void* hnsw_aligned_alloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

/**
 * @brief Free aligned memory
 * @param ptr Pointer to aligned memory
 */
static HNSW_INLINE void hnsw_aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* ========== Time Utilities ========== */

/**
 * @brief Get current timestamp for seeding
 * @return Timestamp value
 */
static HNSW_INLINE unsigned int hnsw_get_seed(void) {
    return (unsigned int)time(NULL);
}

#ifdef __cplusplus
}
#endif

#endif /* HNSW_PLATFORM_H */
