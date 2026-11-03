/**
 * @file hnsw_io.c
 * @brief File I/O implementation for HNSW index persistence
 *
 * Implementation Notes:
 * - Uses binary format for efficiency
 * - Validates file integrity on load
 * - Handles version compatibility
 */

#include "hnsw_io.h"
#include "hnsw_types.h"
#include "distance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========== Internal Helpers ========== */

/**
 * @brief Allocate and initialize neighbors for a node
 */
static int allocate_node_neighbors(HNSWNode* node, int32_t max_level) {
    int32_t total_levels = max_level + 1;

    node->neighbors = (int32_t*)calloc(total_levels * HNSW_M_MAX, sizeof(int32_t));
    if (!node->neighbors) return 0;

    node->neighbor_counts = (int32_t*)calloc(total_levels, sizeof(int32_t));
    if (!node->neighbor_counts) {
        free(node->neighbors);
        node->neighbors = NULL;
        return 0;
    }

    /* Initialize neighbor IDs to -1 (invalid) */
    for (int32_t i = 0; i < total_levels * HNSW_M_MAX; i++) {
        node->neighbors[i] = -1;
    }

    node->max_level = max_level;
    return 1;
}

/**
 * @brief Free node resources
 */
static void free_node(HNSWNode* node) {
    if (node) {
        free(node->vector);
        free(node->neighbors);
        free(node->neighbor_counts);
        node->vector = NULL;
        node->neighbors = NULL;
        node->neighbor_counts = NULL;
    }
}

/* ========== Public Functions ========== */

int hnsw_io_save(const struct HNSWIndex* index, const char* filepath) {
    if (!index || !filepath) {
        return HNSW_ERR_INVALID_PARAM;
    }

    if (index->node_count == 0) {
        return HNSW_ERR_EMPTY_INDEX;
    }

    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        return HNSW_ERR_FILE_IO;
    }

    /* Prepare header */
    HNSWFileHeader header = {0};
    header.magic = HNSW_FILE_MAGIC;
    header.version = HNSW_FILE_VERSION;
    header.dim = index->dim;
    header.node_count = index->node_count;
    header.max_level = index->max_level;
    header.entry_point = index->entry_point;
    header.distance_type = (int32_t)index->distance_type;

    /* Write header */
    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return HNSW_ERR_FILE_IO;
    }

    /* Write nodes */
    for (int32_t i = 0; i < index->node_count; i++) {
        const HNSWNode* node = &index->nodes[i];

        /* Write vector */
        if (fwrite(node->vector, sizeof(float), index->dim, fp) != (size_t)index->dim) {
            fclose(fp);
            return HNSW_ERR_FILE_IO;
        }

        /* Write max level */
        if (fwrite(&node->max_level, sizeof(int32_t), 1, fp) != 1) {
            fclose(fp);
            return HNSW_ERR_FILE_IO;
        }

        /* Write neighbors for each level */
        for (int32_t l = 0; l <= node->max_level; l++) {
            int32_t count = node->neighbor_counts[l];

            if (fwrite(&count, sizeof(int32_t), 1, fp) != 1) {
                fclose(fp);
                return HNSW_ERR_FILE_IO;
            }

            if (count > 0) {
                int32_t* level_neighbors = &node->neighbors[l * HNSW_M_MAX];
                if (fwrite(level_neighbors, sizeof(int32_t), count, fp) != (size_t)count) {
                    fclose(fp);
                    return HNSW_ERR_FILE_IO;
                }
            }
        }
    }

    fclose(fp);
    return HNSW_OK;
}

int hnsw_io_load(const char* filepath, struct HNSWIndex** out_index) {
    if (!filepath || !out_index) {
        return HNSW_ERR_INVALID_PARAM;
    }

    *out_index = NULL;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        return HNSW_ERR_FILE_IO;
    }

    /* Read and validate header */
    HNSWFileHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return HNSW_ERR_FILE_IO;
    }

    if (header.magic != HNSW_FILE_MAGIC) {
        fclose(fp);
        return HNSW_ERR_INVALID_FILE;
    }

    if (header.version < HNSW_FILE_MIN_VERSION || header.version > HNSW_FILE_VERSION) {
        fclose(fp);
        return HNSW_ERR_VERSION_MISMATCH;
    }

    /* Allocate index */
    struct HNSWIndex* index = (struct HNSWIndex*)calloc(1, sizeof(struct HNSWIndex));
    if (!index) {
        fclose(fp);
        return HNSW_ERR_MEMORY;
    }

    /* Initialize index properties */
    index->dim = header.dim;
    index->node_count = 0;  /* Will increment as we load */
    index->node_capacity = header.node_count;
    index->max_level = header.max_level;
    index->entry_point = header.entry_point;
    index->distance_type = (HNSWDistanceType)header.distance_type;
    index->distance_func = distance_get_function(index->distance_type);
    index->is_modified = 0;
    index->random_seed = 0;

    /* Allocate nodes array */
    index->nodes = (HNSWNode*)calloc(header.node_count, sizeof(HNSWNode));
    if (!index->nodes) {
        free(index);
        fclose(fp);
        return HNSW_ERR_MEMORY;
    }

    /* Read nodes */
    for (int32_t i = 0; i < header.node_count; i++) {
        HNSWNode* node = &index->nodes[i];
        node->id = i;

        /* Read vector */
        node->vector = (float*)malloc(header.dim * sizeof(float));
        if (!node->vector) {
            goto cleanup_error;
        }

        if (fread(node->vector, sizeof(float), header.dim, fp) != (size_t)header.dim) {
            goto cleanup_error;
        }

        /* Read max level */
        int32_t node_max_level;
        if (fread(&node_max_level, sizeof(int32_t), 1, fp) != 1) {
            goto cleanup_error;
        }

        /* Allocate neighbors */
        if (!allocate_node_neighbors(node, node_max_level)) {
            goto cleanup_error;
        }

        /* Read neighbors for each level */
        for (int32_t l = 0; l <= node_max_level; l++) {
            int32_t count;
            if (fread(&count, sizeof(int32_t), 1, fp) != 1) {
                goto cleanup_error;
            }

            node->neighbor_counts[l] = count;

            if (count > 0) {
                int32_t* level_neighbors = &node->neighbors[l * HNSW_M_MAX];
                if (fread(level_neighbors, sizeof(int32_t), count, fp) != (size_t)count) {
                    goto cleanup_error;
                }
            }
        }

        index->node_count++;
    }

    fclose(fp);
    *out_index = index;
    return HNSW_OK;

cleanup_error:
    /* Free all loaded nodes */
    for (int32_t i = 0; i < index->node_count; i++) {
        free_node(&index->nodes[i]);
    }
    /* Free the node being loaded if partially allocated */
    if (index->node_count < header.node_count) {
        free_node(&index->nodes[index->node_count]);
    }
    free(index->nodes);
    free(index);
    fclose(fp);
    return HNSW_ERR_FILE_IO;
}

int hnsw_io_validate(const char* filepath, HNSWFileHeader* out_header) {
    if (!filepath) {
        return HNSW_ERR_INVALID_PARAM;
    }

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        return HNSW_ERR_FILE_IO;
    }

    HNSWFileHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return HNSW_ERR_FILE_IO;
    }

    fclose(fp);

    if (header.magic != HNSW_FILE_MAGIC) {
        return HNSW_ERR_INVALID_FILE;
    }

    if (header.version < HNSW_FILE_MIN_VERSION || header.version > HNSW_FILE_VERSION) {
        return HNSW_ERR_VERSION_MISMATCH;
    }

    if (out_header) {
        *out_header = header;
    }

    return HNSW_OK;
}

int hnsw_io_info(const char* filepath, int32_t* out_dim, int32_t* out_count) {
    HNSWFileHeader header;
    int result = hnsw_io_validate(filepath, &header);

    if (result != HNSW_OK) {
        return result;
    }

    if (out_dim) {
        *out_dim = header.dim;
    }
    if (out_count) {
        *out_count = header.node_count;
    }

    return HNSW_OK;
}
