"""
CFFI build script for the HNSW C core.

Compiles the modular C sources into a Python-accessible extension module
(hnswdb._hnsw_cffi). Each module is compiled as its own translation unit:
- Core HNSW algorithm (hnsw.c)
- Priority queue      (priority_queue.c)
- Distance functions  (distance.c)
- File I/O            (hnsw_io.c)

Usage (from the repository root):
    python hnswdb/build_cffi.py
or simply:
    pip install .
"""

import os

from cffi import FFI

ffi = FFI()

# Public C interface exposed to Python (mirrors core/hnsw.h)
ffi.cdef("""
    /* Error codes */
    #define HNSW_OK                     0
    #define HNSW_ERR_MEMORY            -1
    #define HNSW_ERR_INVALID_PARAM     -2
    #define HNSW_ERR_NOT_INITIALIZED   -3
    #define HNSW_ERR_FILE_IO           -4
    #define HNSW_ERR_DIMENSION_MISMATCH -5
    #define HNSW_ERR_EMPTY_INDEX       -6
    #define HNSW_ERR_INVALID_FILE      -7
    #define HNSW_ERR_VERSION_MISMATCH  -8

    /* Opaque handle */
    typedef struct HNSWIndex HNSWIndex;

    /* Result structure */
    typedef struct {
        int32_t id;
        float distance;
    } HNSWResult;

    /* Lifecycle */
    HNSWIndex* hnsw_create(int32_t dim);
    void hnsw_destroy(HNSWIndex* index);

    /* Vector operations */
    int hnsw_add(HNSWIndex* index, const float* vectors, int32_t n, int32_t dim);
    int hnsw_search(HNSWIndex* index, const float* query, int32_t dim,
                    int32_t k, HNSWResult* results, int32_t* num_results);
    int hnsw_get_vectors(HNSWIndex* index, float* vectors, int32_t* count, int32_t* dim);

    /* Persistence */
    int hnsw_save(HNSWIndex* index, const char* filepath);
    HNSWIndex* hnsw_load(const char* filepath);

    /* Properties */
    int32_t hnsw_count(HNSWIndex* index);
    int32_t hnsw_dimension(HNSWIndex* index);
    int hnsw_is_modified(HNSWIndex* index);
    void hnsw_mark_saved(HNSWIndex* index);

    /* Error handling */
    const char* hnsw_error_string(int error_code);
""")

# Locate the C sources
source_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(source_dir, "core")


def _rel(path):
    """setuptools prefers relative source paths; fall back to absolute if impossible."""
    try:
        return os.path.relpath(path)
    except ValueError:  # e.g. different drive on Windows
        return path


c_sources = [
    _rel(os.path.join(core_dir, name))
    for name in ("priority_queue.c", "distance.c", "hnsw_io.c", "hnsw.c")
]

# Compile flags based on platform
if os.name == "nt":
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O3", "-std=gnu99"]

ffi.set_source(
    "hnswdb._hnsw_cffi",
    '#include "hnsw.h"',
    sources=c_sources,
    include_dirs=[_rel(core_dir)],
    define_macros=[("HNSW_EXPORTS", "1")],
    extra_compile_args=extra_compile_args,
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
