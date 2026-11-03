/**
 * @file hnsw.c
 * @brief HNSW Index Core Implementation
 *
 * This file implements the core HNSW algorithm for approximate nearest
 * neighbor search. The implementation follows the paper:
 * "Efficient and robust approximate nearest neighbor search using
 * Hierarchical Navigable Small World graphs" by Malkov & Yashunin
 *
 * Design Principles:
 * - Single Responsibility: Core graph operations only
 * - Dependency Injection: Distance function via pointer
 * - Separation of Concerns: I/O, distance, queue in separate modules
 *
 * Algorithm Overview:
 * 1. Multi-layer graph with probabilistic level assignment
 * 2. Greedy search from top layer, increasingly refined
 * 3. Bidirectional connections for graph connectivity
 */

#include "hnsw.h"
#include "hnsw_types.h"
#include "priority_queue.h"
#include "distance.h"
#include "hnsw_io.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ========== Visited Set (Bitmap) ========== */

/**
 * @struct VisitedSet
 * @brief Efficient set for tracking visited nodes using bitmap
 */
typedef struct {
    uint32_t* bits;
    int32_t   capacity;
} VisitedSet;

/**
 * @brief Create a visited set
 */
static VisitedSet* visited_create(int32_t max_nodes) {
    VisitedSet* vs = (VisitedSet*)malloc(sizeof(VisitedSet));
    if (!vs) return NULL;

    vs->capacity = (max_nodes + 31) / 32;
    vs->bits = (uint32_t*)calloc(vs->capacity, sizeof(uint32_t));
    if (!vs->bits) {
        free(vs);
        return NULL;
    }

    return vs;
}

/**
 * @brief Destroy a visited set
 */
static void visited_destroy(VisitedSet* vs) {
    if (vs) {
        free(vs->bits);
        free(vs);
    }
}

/**
 * @brief Mark a node as visited
 */
static HNSW_INLINE void visited_set(VisitedSet* vs, int32_t id) {
    vs->bits[id / 32] |= (1u << (id % 32));
}

/**
 * @brief Check if a node was visited
 */
static HNSW_INLINE int visited_get(const VisitedSet* vs, int32_t id) {
    return (vs->bits[id / 32] & (1u << (id % 32))) != 0;
}

/* ========== Node Management ========== */

/**
 * @brief Allocate neighbors for a node at creation
 */
static int node_allocate_neighbors(HNSWNode* node, int32_t max_level) {
    int32_t total_levels = max_level + 1;
    int32_t total_slots = total_levels * HNSW_M_MAX;

    node->neighbors = (int32_t*)malloc(total_slots * sizeof(int32_t));
    if (!node->neighbors) return 0;

    node->neighbor_counts = (int32_t*)calloc(total_levels, sizeof(int32_t));
    if (!node->neighbor_counts) {
        free(node->neighbors);
        node->neighbors = NULL;
        return 0;
    }

    /* Initialize all neighbor slots to -1 (invalid) */
    for (int32_t i = 0; i < total_slots; i++) {
        node->neighbors[i] = -1;
    }

    node->max_level = max_level;
    return 1;
}

/**
 * @brief Free node resources
 */
static void node_free(HNSWNode* node) {
    if (node) {
        free(node->vector);
        free(node->neighbors);
        free(node->neighbor_counts);
        memset(node, 0, sizeof(HNSWNode));
    }
}

/**
 * @brief Get pointer to neighbors at a level
 */
static HNSW_INLINE int32_t* node_get_neighbors(HNSWNode* node, int32_t level,
                                                int32_t* out_count) {
    if (level > node->max_level) {
        *out_count = 0;
        return NULL;
    }
    *out_count = node->neighbor_counts[level];
    return &node->neighbors[level * HNSW_M_MAX];
}

/**
 * @brief Add a neighbor to a node (if room available)
 */
static void node_add_neighbor(HNSWNode* node, int32_t level, int32_t neighbor_id) {
    int32_t max_neighbors = HNSW_MAX_NEIGHBORS(level);
    int32_t count = node->neighbor_counts[level];

    if (count < max_neighbors) {
        node->neighbors[level * HNSW_M_MAX + count] = neighbor_id;
        node->neighbor_counts[level]++;
    }
}

/* ========== Level Generation ========== */

/**
 * @brief Generate random level for new node
 *
 * Uses exponential distribution: P(level=l) = (1/M)^l * (1 - 1/M)
 * Expected max level: log_M(n) where n = number of nodes
 */
static int32_t generate_random_level(HNSWIndex* index) {
    float r = (float)hnsw_rand_r(&index->random_seed) / (float)RAND_MAX;

    /* Avoid log(0) */
    if (r < 1e-9f) r = 1e-9f;

    return (int32_t)(-logf(r) * HNSW_LEVEL_MULTIPLIER);
}

/* ========== Graph Search ========== */

/**
 * @brief Search a single layer for nearest neighbors (convergent search)
 *
 * Best-first search that keeps expanding while any candidate could still
 * improve the current top-k. No ef parameter.
 *
 * @param index      HNSW index
 * @param query      Query vector
 * @param entry_id   Starting node ID
 * @param level      Layer to search
 * @param k          Number of results to return (0 = find single best)
 * @return Priority queue of results (max-heap by distance)
 */
static PriorityQueue* search_layer(HNSWIndex* index, const float* query,
                                    int32_t entry_id, int32_t level, int32_t k) {
    /* Create visited set */
    VisitedSet* visited = visited_create(index->node_count);
    if (!visited) return NULL;

    /* k=0 means find the single closest */
    int32_t target_k = (k <= 0) ? 1 : k;

    /* Calculate entry point distance */
    float entry_dist = index->distance_func(query, index->nodes[entry_id].vector,
                                             index->dim);

    /* Create candidate queue (min-heap) and result queue (max-heap) */
    int32_t initial_capacity = HNSW_MAX(target_k * 4, 64);
    PriorityQueue* candidates = pq_create(initial_capacity, PQ_MIN_HEAP);
    PriorityQueue* results = pq_create(initial_capacity, PQ_MAX_HEAP);

    if (!candidates || !results) {
        pq_destroy(candidates);
        pq_destroy(results);
        visited_destroy(visited);
        return NULL;
    }

    /* Initialize with entry point */
    pq_push(candidates, entry_id, entry_dist);
    pq_push(results, entry_id, entry_dist);
    visited_set(visited, entry_id);

    /* Convergent search - continue until no candidate can improve the top-k */
    while (!pq_is_empty(candidates)) {
        PQNode current = pq_pop(candidates);

        /* Once top-k is full, stop when the closest remaining candidate is
         * farther than the worst result: nothing reachable can improve it.
         * While top-k is not yet full, always keep expanding. */
        if (pq_size(results) >= target_k &&
            current.distance > pq_top(results).distance) {
            break;
        }

        /* Explore neighbors */
        HNSWNode* node = &index->nodes[current.id];
        int32_t neighbor_count;
        int32_t* neighbors = node_get_neighbors(node, level, &neighbor_count);

        for (int32_t i = 0; i < neighbor_count; i++) {
            int32_t neighbor_id = neighbors[i];

            /* Skip invalid or visited */
            if (neighbor_id < 0 || visited_get(visited, neighbor_id)) {
                continue;
            }

            visited_set(visited, neighbor_id);

            /* Compute distance */
            float dist = index->distance_func(query,
                                               index->nodes[neighbor_id].vector,
                                               index->dim);

            /* Add to results if promising */
            if (pq_size(results) < target_k) {
                /* Results not full - always add */
                pq_push(candidates, neighbor_id, dist);
                pq_push(results, neighbor_id, dist);
            } else {
                /* Only add if better than worst result */
                PQNode worst = pq_top(results);
                if (dist < worst.distance) {
                    pq_push(candidates, neighbor_id, dist);
                    pq_push(results, neighbor_id, dist);
                    pq_pop(results);  /* Remove worst */
                }
            }
        }
    }

    pq_destroy(candidates);
    visited_destroy(visited);

    return results;
}

/**
 * @brief Select M best neighbors from candidates
 *
 * Simple selection heuristic: take M closest.
 * Could be extended to use diversity heuristic.
 */
static void select_neighbors(HNSWIndex* index, int32_t node_id,
                              PriorityQueue* candidates, int32_t level) {
    HNSWNode* node = &index->nodes[node_id];
    int32_t M = HNSW_MAX_NEIGHBORS(level);

    int32_t count;
    const PQNode* data = pq_data(candidates, &count);

    /* Collect all candidates into array for sorting */
    PQNode* sorted = (PQNode*)malloc(count * sizeof(PQNode));
    if (!sorted) return;

    memcpy(sorted, data, count * sizeof(PQNode));

    /* Sort by distance (simple O(n^2) sort - candidate lists are small) */
    for (int32_t i = 0; i < count - 1; i++) {
        for (int32_t j = i + 1; j < count; j++) {
            if (sorted[j].distance < sorted[i].distance) {
                PQNode temp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = temp;
            }
        }
    }

    /* Add M closest as neighbors */
    int32_t to_add = HNSW_MIN(M, count);
    for (int32_t i = 0; i < to_add; i++) {
        node_add_neighbor(node, level, sorted[i].id);
    }

    free(sorted);
}

/**
 * @brief Create bidirectional connections with neighbors
 */
static void connect_neighbors(HNSWIndex* index, int32_t node_id, int32_t level) {
    HNSWNode* node = &index->nodes[node_id];
    int32_t neighbor_count;
    int32_t* neighbors = node_get_neighbors(node, level, &neighbor_count);
    int32_t M = HNSW_MAX_NEIGHBORS(level);

    for (int32_t i = 0; i < neighbor_count; i++) {
        int32_t neighbor_id = neighbors[i];
        if (neighbor_id < 0) continue;

        HNSWNode* neighbor = &index->nodes[neighbor_id];

        /* Check if neighbor has room for connection */
        int32_t n_count;
        int32_t* n_neighbors = node_get_neighbors(neighbor, level, &n_count);

        if (n_count < M) {
            /* Room available - add connection */
            node_add_neighbor(neighbor, level, node_id);
        } else {
            /* No room - check if we should replace worst neighbor */
            float new_dist = index->distance_func(node->vector, neighbor->vector,
                                                   index->dim);

            int32_t worst_idx = -1;
            float worst_dist = new_dist;

            for (int32_t j = 0; j < n_count; j++) {
                if (n_neighbors[j] < 0) continue;

                float dist = index->distance_func(neighbor->vector,
                                                   index->nodes[n_neighbors[j]].vector,
                                                   index->dim);
                if (dist > worst_dist) {
                    worst_dist = dist;
                    worst_idx = j;
                }
            }

            if (worst_idx >= 0) {
                n_neighbors[worst_idx] = node_id;
            }
        }
    }
}

/* ========== Public API ========== */

HNSW_API HNSWIndex* hnsw_create(int32_t dim) {
    HNSWIndex* index = (HNSWIndex*)calloc(1, sizeof(HNSWIndex));
    if (!index) return NULL;

    /* Initialize properties */
    index->dim = dim;
    index->node_count = 0;
    index->node_capacity = HNSW_INITIAL_CAPACITY;
    index->entry_point = -1;
    index->max_level = -1;

    /* Default to cosine distance */
    index->distance_type = HNSW_DISTANCE_COSINE;
    index->distance_func = distance_get_function(HNSW_DISTANCE_COSINE);

    /* State */
    index->is_modified = 0;
    index->random_seed = hnsw_get_seed();

    /* Allocate nodes array */
    index->nodes = (HNSWNode*)calloc(HNSW_INITIAL_CAPACITY, sizeof(HNSWNode));
    if (!index->nodes) {
        free(index);
        return NULL;
    }

    return index;
}

HNSW_API void hnsw_destroy(HNSWIndex* index) {
    if (!index) return;

    /* Free all nodes */
    for (int32_t i = 0; i < index->node_count; i++) {
        node_free(&index->nodes[i]);
    }

    free(index->nodes);
    free(index);
}

HNSW_API int hnsw_add(HNSWIndex* index, const float* vectors, int32_t n, int32_t dim) {
    /* Validate inputs */
    if (!index || !vectors || n <= 0 || dim <= 0) {
        return HNSW_ERR_INVALID_PARAM;
    }

    /* Auto-detect or validate dimension */
    if (index->dim == 0) {
        index->dim = dim;
    } else if (index->dim != dim) {
        return HNSW_ERR_DIMENSION_MISMATCH;
    }

    /* Ensure capacity */
    if (index->node_count + n > index->node_capacity) {
        int32_t new_capacity = (index->node_count + n) * 2;
        HNSWNode* new_nodes = (HNSWNode*)realloc(index->nodes,
                                                   new_capacity * sizeof(HNSWNode));
        if (!new_nodes) {
            return HNSW_ERR_MEMORY;
        }

        /* Zero new memory */
        memset(&new_nodes[index->node_capacity], 0,
               (new_capacity - index->node_capacity) * sizeof(HNSWNode));

        index->nodes = new_nodes;
        index->node_capacity = new_capacity;
    }

    /* Insert each vector */
    for (int32_t i = 0; i < n; i++) {
        int32_t node_id = index->node_count;
        HNSWNode* node = &index->nodes[node_id];

        /* Allocate and copy vector */
        node->vector = (float*)malloc(dim * sizeof(float));
        if (!node->vector) {
            return HNSW_ERR_MEMORY;
        }
        memcpy(node->vector, &vectors[i * dim], dim * sizeof(float));

        node->id = node_id;

        /* Generate random level */
        int32_t level = generate_random_level(index);

        /* Allocate neighbor storage */
        if (!node_allocate_neighbors(node, level)) {
            free(node->vector);
            return HNSW_ERR_MEMORY;
        }

        index->node_count++;

        /* First node: just set as entry point */
        if (index->entry_point < 0) {
            index->entry_point = node_id;
            index->max_level = level;
            index->is_modified = 1;
            continue;
        }

        /* Search down from top to find insertion point (greedy single best) */
        int32_t entry_id = index->entry_point;

        for (int32_t l = index->max_level; l > level; l--) {
            PriorityQueue* results = search_layer(index, node->vector, entry_id, l, 1);
            if (results && !pq_is_empty(results)) {
                entry_id = pq_pop(results).id;
            }
            pq_destroy(results);
        }

        /* Insert into each layer from node's level down to 0 */
        /* Search for M neighbors to connect to */
        int32_t insert_level = HNSW_MIN(level, index->max_level);
        for (int32_t l = insert_level; l >= 0; l--) {
            int32_t M = HNSW_MAX_NEIGHBORS(l);
            PriorityQueue* candidates = search_layer(index, node->vector, entry_id, l, M);
            if (candidates) {
                select_neighbors(index, node_id, candidates, l);
                pq_destroy(candidates);
                connect_neighbors(index, node_id, l);
            }
        }

        /* Update entry point if new node has higher level */
        if (level > index->max_level) {
            index->entry_point = node_id;
            index->max_level = level;
        }

        index->is_modified = 1;
    }

    return HNSW_OK;
}

HNSW_API int hnsw_search(HNSWIndex* index, const float* query, int32_t dim,
                          int32_t k, HNSWResult* results, int32_t* num_results) {
    /* Validate inputs */
    if (!index || !query || !results || !num_results || k <= 0) {
        return HNSW_ERR_INVALID_PARAM;
    }

    if (index->node_count == 0) {
        *num_results = 0;
        return HNSW_ERR_EMPTY_INDEX;
    }

    if (dim != index->dim) {
        return HNSW_ERR_DIMENSION_MISMATCH;
    }

    /* Limit k to available nodes */
    k = HNSW_MIN(k, index->node_count);

    /* Search from entry point down to layer 1 (greedy single best per layer) */
    int32_t entry_id = index->entry_point;

    for (int32_t l = index->max_level; l > 0; l--) {
        PriorityQueue* layer_results = search_layer(index, query, entry_id, l, 1);
        if (layer_results && !pq_is_empty(layer_results)) {
            entry_id = pq_pop(layer_results).id;
        }
        pq_destroy(layer_results);
    }

    /* Final search at layer 0 - find k closest (convergent search) */
    PriorityQueue* final = search_layer(index, query, entry_id, 0, k);
    if (!final) {
        *num_results = 0;
        return HNSW_ERR_MEMORY;
    }

    /* Extract and sort results */
    int32_t count = pq_size(final);
    int32_t result_count = HNSW_MIN(k, count);

    /* Results are in max-heap, need to reverse for closest first */
    PQNode* temp = (PQNode*)malloc(count * sizeof(PQNode));
    if (!temp) {
        pq_destroy(final);
        *num_results = 0;
        return HNSW_ERR_MEMORY;
    }

    int32_t n = 0;
    while (!pq_is_empty(final)) {
        temp[n++] = pq_pop(final);
    }

    /* Copy in reverse order (closest first) */
    for (int32_t i = 0; i < result_count; i++) {
        results[i].id = temp[n - 1 - i].id;
        results[i].distance = temp[n - 1 - i].distance;
    }

    free(temp);
    pq_destroy(final);

    *num_results = result_count;
    return HNSW_OK;
}

HNSW_API int hnsw_get_vectors(HNSWIndex* index, float* vectors,
                               int32_t* count, int32_t* dim) {
    if (!index || !count || !dim) {
        return HNSW_ERR_INVALID_PARAM;
    }

    *count = index->node_count;
    *dim = index->dim;

    if (vectors && index->node_count > 0) {
        for (int32_t i = 0; i < index->node_count; i++) {
            memcpy(&vectors[i * index->dim], index->nodes[i].vector,
                   index->dim * sizeof(float));
        }
    }

    return HNSW_OK;
}

HNSW_API int hnsw_save(HNSWIndex* index, const char* filepath) {
    int result = hnsw_io_save(index, filepath);
    if (result == HNSW_OK) {
        index->is_modified = 0;
    }
    return result;
}

HNSW_API HNSWIndex* hnsw_load(const char* filepath) {
    HNSWIndex* index = NULL;
    int result = hnsw_io_load(filepath, &index);

    if (result != HNSW_OK) {
        return NULL;
    }

    /* Reinitialize runtime fields */
    index->random_seed = hnsw_get_seed();

    return index;
}

HNSW_API int32_t hnsw_count(HNSWIndex* index) {
    return index ? index->node_count : -1;
}

HNSW_API int32_t hnsw_dimension(HNSWIndex* index) {
    return index ? index->dim : -1;
}

HNSW_API int hnsw_is_modified(HNSWIndex* index) {
    return index ? index->is_modified : -1;
}

HNSW_API void hnsw_mark_saved(HNSWIndex* index) {
    if (index) {
        index->is_modified = 0;
    }
}

HNSW_API const char* hnsw_error_string(int error_code) {
    switch (error_code) {
        case HNSW_OK:
            return "Success";
        case HNSW_ERR_MEMORY:
            return "Memory allocation failed";
        case HNSW_ERR_INVALID_PARAM:
            return "Invalid parameter";
        case HNSW_ERR_NOT_INITIALIZED:
            return "Index not initialized";
        case HNSW_ERR_FILE_IO:
            return "File I/O error";
        case HNSW_ERR_DIMENSION_MISMATCH:
            return "Vector dimension mismatch";
        case HNSW_ERR_EMPTY_INDEX:
            return "Index is empty";
        case HNSW_ERR_INVALID_FILE:
            return "Invalid file format";
        case HNSW_ERR_VERSION_MISMATCH:
            return "File version not supported";
        default:
            return "Unknown error";
    }
}
