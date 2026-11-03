/**
 * @file priority_queue.h
 * @brief Priority queue (binary heap) for HNSW candidate management
 *
 * This module provides a generic priority queue implementation used
 * for managing candidates during HNSW graph traversal.
 *
 * Design Principles:
 * - Single Responsibility: Only handles priority queue operations
 * - Interface Segregation: Clean, minimal API
 * - Encapsulation: Internal heap structure hidden
 *
 * Usage:
 *   PriorityQueue* pq = pq_create(100, PQ_MIN_HEAP);
 *   pq_push(pq, 42, 0.5f);
 *   PQNode top = pq_top(pq);
 *   PQNode popped = pq_pop(pq);
 *   pq_destroy(pq);
 */

#ifndef HNSW_PRIORITY_QUEUE_H
#define HNSW_PRIORITY_QUEUE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Types ========== */

/**
 * @enum PQHeapType
 * @brief Specifies heap ordering
 */
typedef enum {
    PQ_MIN_HEAP = 1,  /**< Smallest element at top */
    PQ_MAX_HEAP = 0   /**< Largest element at top */
} PQHeapType;

/**
 * @struct PQNode
 * @brief Element stored in the priority queue
 */
typedef struct {
    int32_t id;       /**< Item identifier */
    float   distance; /**< Priority value */
} PQNode;

/**
 * @struct PriorityQueue
 * @brief Opaque priority queue handle
 */
typedef struct PriorityQueue PriorityQueue;

/* ========== Lifecycle ========== */

/**
 * @brief Create a new priority queue
 *
 * @param initial_capacity Initial capacity (will grow as needed)
 * @param heap_type PQ_MIN_HEAP or PQ_MAX_HEAP
 * @return Pointer to queue or NULL on allocation failure
 */
PriorityQueue* pq_create(int32_t initial_capacity, PQHeapType heap_type);

/**
 * @brief Destroy a priority queue and free memory
 *
 * @param pq Queue to destroy (NULL-safe)
 */
void pq_destroy(PriorityQueue* pq);

/**
 * @brief Clear all elements without deallocating
 *
 * @param pq Queue to clear
 */
void pq_clear(PriorityQueue* pq);

/* ========== Operations ========== */

/**
 * @brief Push an element onto the queue
 *
 * @param pq Queue
 * @param id Item identifier
 * @param distance Priority value
 * @return true on success, false on allocation failure
 */
bool pq_push(PriorityQueue* pq, int32_t id, float distance);

/**
 * @brief Remove and return the top element
 *
 * @param pq Queue (must not be empty)
 * @return The top element
 *
 * @pre !pq_is_empty(pq)
 */
PQNode pq_pop(PriorityQueue* pq);

/**
 * @brief Get the top element without removing
 *
 * @param pq Queue (must not be empty)
 * @return The top element
 *
 * @pre !pq_is_empty(pq)
 */
PQNode pq_top(const PriorityQueue* pq);

/* ========== Queries ========== */

/**
 * @brief Check if queue is empty
 *
 * @param pq Queue
 * @return true if empty
 */
bool pq_is_empty(const PriorityQueue* pq);

/**
 * @brief Get number of elements
 *
 * @param pq Queue
 * @return Element count
 */
int32_t pq_size(const PriorityQueue* pq);

/**
 * @brief Get current capacity
 *
 * @param pq Queue
 * @return Capacity
 */
int32_t pq_capacity(const PriorityQueue* pq);

/* ========== Bulk Operations ========== */

/**
 * @brief Reserve capacity for at least n elements
 *
 * @param pq Queue
 * @param n Minimum capacity
 * @return true on success
 */
bool pq_reserve(PriorityQueue* pq, int32_t n);

/**
 * @brief Get all elements as array (heap order, not sorted)
 *
 * @param pq Queue
 * @param out_count Output: number of elements
 * @return Pointer to internal array (valid until next modification)
 */
const PQNode* pq_data(const PriorityQueue* pq, int32_t* out_count);

#ifdef __cplusplus
}
#endif

#endif /* HNSW_PRIORITY_QUEUE_H */
