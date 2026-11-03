/**
 * @file priority_queue.c
 * @brief Priority queue implementation using binary heap
 *
 * Implementation Details:
 * - Binary heap stored in dynamic array
 * - Automatic capacity growth (2x)
 * - O(log n) push and pop
 * - O(1) top and size queries
 */

#include "priority_queue.h"
#include "platform.h"
#include <stdlib.h>
#include <string.h>

/* ========== Internal Structure ========== */

/**
 * @struct PriorityQueue
 * @brief Binary heap implementation
 */
struct PriorityQueue {
    PQNode*    nodes;       /**< Heap array */
    int32_t    size;        /**< Current number of elements */
    int32_t    capacity;    /**< Allocated capacity */
    PQHeapType heap_type;   /**< Min or max heap */
};

/* ========== Private Functions ========== */

/**
 * @brief Compare two elements based on heap type
 * @return true if nodes[i] should be above nodes[j]
 */
static HNSW_INLINE bool pq_should_swap(const PriorityQueue* pq, int32_t i, int32_t j) {
    if (pq->heap_type == PQ_MIN_HEAP) {
        return pq->nodes[i].distance < pq->nodes[j].distance;
    } else {
        return pq->nodes[i].distance > pq->nodes[j].distance;
    }
}

/**
 * @brief Swap two elements
 */
static HNSW_INLINE void pq_swap(PriorityQueue* pq, int32_t i, int32_t j) {
    PQNode temp = pq->nodes[i];
    pq->nodes[i] = pq->nodes[j];
    pq->nodes[j] = temp;
}

/**
 * @brief Restore heap property upward from index
 */
static void pq_sift_up(PriorityQueue* pq, int32_t index) {
    while (index > 0) {
        int32_t parent = (index - 1) / 2;
        if (pq_should_swap(pq, index, parent)) {
            pq_swap(pq, index, parent);
            index = parent;
        } else {
            break;
        }
    }
}

/**
 * @brief Restore heap property downward from index
 */
static void pq_sift_down(PriorityQueue* pq, int32_t index) {
    while (true) {
        int32_t best = index;
        int32_t left = 2 * index + 1;
        int32_t right = 2 * index + 2;

        if (left < pq->size && pq_should_swap(pq, left, best)) {
            best = left;
        }
        if (right < pq->size && pq_should_swap(pq, right, best)) {
            best = right;
        }

        if (best != index) {
            pq_swap(pq, index, best);
            index = best;
        } else {
            break;
        }
    }
}

/**
 * @brief Grow capacity by 2x
 */
static bool pq_grow(PriorityQueue* pq) {
    int32_t new_capacity = pq->capacity * 2;
    if (new_capacity < 16) new_capacity = 16;

    PQNode* new_nodes = (PQNode*)realloc(pq->nodes, new_capacity * sizeof(PQNode));
    if (!new_nodes) {
        return false;
    }

    pq->nodes = new_nodes;
    pq->capacity = new_capacity;
    return true;
}

/* ========== Public API ========== */

PriorityQueue* pq_create(int32_t initial_capacity, PQHeapType heap_type) {
    if (initial_capacity <= 0) {
        initial_capacity = 16;
    }

    PriorityQueue* pq = (PriorityQueue*)malloc(sizeof(PriorityQueue));
    if (!pq) {
        return NULL;
    }

    pq->nodes = (PQNode*)malloc(initial_capacity * sizeof(PQNode));
    if (!pq->nodes) {
        free(pq);
        return NULL;
    }

    pq->size = 0;
    pq->capacity = initial_capacity;
    pq->heap_type = heap_type;

    return pq;
}

void pq_destroy(PriorityQueue* pq) {
    if (pq) {
        free(pq->nodes);
        free(pq);
    }
}

void pq_clear(PriorityQueue* pq) {
    if (pq) {
        pq->size = 0;
    }
}

bool pq_push(PriorityQueue* pq, int32_t id, float distance) {
    if (!pq) {
        return false;
    }

    /* Grow if needed */
    if (pq->size >= pq->capacity) {
        if (!pq_grow(pq)) {
            return false;
        }
    }

    /* Insert at end and sift up */
    pq->nodes[pq->size].id = id;
    pq->nodes[pq->size].distance = distance;
    pq_sift_up(pq, pq->size);
    pq->size++;

    return true;
}

PQNode pq_pop(PriorityQueue* pq) {
    PQNode result = pq->nodes[0];

    pq->size--;
    if (pq->size > 0) {
        pq->nodes[0] = pq->nodes[pq->size];
        pq_sift_down(pq, 0);
    }

    return result;
}

PQNode pq_top(const PriorityQueue* pq) {
    return pq->nodes[0];
}

bool pq_is_empty(const PriorityQueue* pq) {
    return pq == NULL || pq->size == 0;
}

int32_t pq_size(const PriorityQueue* pq) {
    return pq ? pq->size : 0;
}

int32_t pq_capacity(const PriorityQueue* pq) {
    return pq ? pq->capacity : 0;
}

bool pq_reserve(PriorityQueue* pq, int32_t n) {
    if (!pq || n <= pq->capacity) {
        return true;
    }

    PQNode* new_nodes = (PQNode*)realloc(pq->nodes, n * sizeof(PQNode));
    if (!new_nodes) {
        return false;
    }

    pq->nodes = new_nodes;
    pq->capacity = n;
    return true;
}

const PQNode* pq_data(const PriorityQueue* pq, int32_t* out_count) {
    if (out_count) {
        *out_count = pq ? pq->size : 0;
    }
    return pq ? pq->nodes : NULL;
}
