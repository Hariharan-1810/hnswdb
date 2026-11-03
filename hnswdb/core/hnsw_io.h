/**
 * @file hnsw_io.h
 * @brief File I/O operations for HNSW index persistence
 *
 * This module handles saving and loading HNSW indices to/from disk.
 *
 * Design Principles:
 * - Single Responsibility: Only handles serialization/deserialization
 * - Validated File Format: Magic number and version checked on load
 * - Forward Compatibility: Version field + reserved header space
 *
 * File Format (v2):
 *   Header:
 *     - Magic number (4 bytes): "HNSW"
 *     - Version (4 bytes): Format version
 *     - Dimension (4 bytes): Vector dimension
 *     - Node count (4 bytes): Number of vectors
 *     - Max level (4 bytes): Maximum graph level
 *     - Entry point (4 bytes): Entry node ID
 *     - Distance type (4 bytes): Distance metric used
 *
 *   For each node:
 *     - Vector data (dim * 4 bytes)
 *     - Max level (4 bytes)
 *     - For each level 0..max_level:
 *       - Neighbor count (4 bytes)
 *       - Neighbor IDs (count * 4 bytes)
 */

#ifndef HNSW_IO_H
#define HNSW_IO_H

#include <stdint.h>
#include "hnsw_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========== File Format Constants ========== */

/** Magic number "HNSW" in little-endian */
#define HNSW_FILE_MAGIC     0x57534E48

/** Current file format version */
#define HNSW_FILE_VERSION   2

/** Minimum supported version for reading */
#define HNSW_FILE_MIN_VERSION 1

/* ========== File Header Structure ========== */

/**
 * @struct HNSWFileHeader
 * @brief Header stored at beginning of HNSW files
 */
typedef struct {
    uint32_t magic;           /**< Magic number for file validation */
    uint32_t version;         /**< File format version */
    int32_t  dim;             /**< Vector dimension */
    int32_t  node_count;      /**< Number of nodes */
    int32_t  max_level;       /**< Maximum level in graph */
    int32_t  entry_point;     /**< Entry point node ID */
    int32_t  distance_type;   /**< Distance metric type */
    int32_t  reserved[5];     /**< Reserved for future use */
} HNSWFileHeader;

/* ========== I/O Functions ========== */

/**
 * @brief Save HNSW index to file
 *
 * @param index Index to save
 * @param filepath Path to output file
 * @return HNSW_OK on success, error code otherwise
 *
 * @note Overwrites the file in place; an interrupted write can leave a
 *       truncated file (atomic temp-file + rename is a planned improvement)
 */
int hnsw_io_save(const struct HNSWIndex* index, const char* filepath);

/**
 * @brief Load HNSW index from file
 *
 * @param filepath Path to input file
 * @param out_index Output: loaded index (caller must destroy)
 * @return HNSW_OK on success, error code otherwise
 */
int hnsw_io_load(const char* filepath, struct HNSWIndex** out_index);

/**
 * @brief Validate file header without loading
 *
 * @param filepath Path to file
 * @param out_header Output: file header if valid
 * @return HNSW_OK if valid, error code otherwise
 */
int hnsw_io_validate(const char* filepath, HNSWFileHeader* out_header);

/**
 * @brief Get file info without loading full index
 *
 * @param filepath Path to file
 * @param out_dim Output: vector dimension
 * @param out_count Output: number of vectors
 * @return HNSW_OK on success, error code otherwise
 */
int hnsw_io_info(const char* filepath, int32_t* out_dim, int32_t* out_count);

#ifdef __cplusplus
}
#endif

#endif /* HNSW_IO_H */
