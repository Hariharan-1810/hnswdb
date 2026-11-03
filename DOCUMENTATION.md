# HNSWDB - Complete Technical Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [HNSW Algorithm Explained](#3-hnsw-algorithm-explained)
4. [C Core Implementation](#4-c-core-implementation)
5. [Python API Reference](#5-python-api-reference)
6. [File Format Specification](#6-file-format-specification)
7. [Build System](#7-build-system)
8. [Configuration & Constants](#8-configuration--constants)
9. [Error Handling](#9-error-handling)
10. [Performance Characteristics](#10-performance-characteristics)
11. [Limitations & Future Work](#11-limitations--future-work)

---

## 1. Project Overview

### What is HNSWDB?

HNSWDB is a **local vector database** that stores and retrieves embeddings using the HNSW (Hierarchical Navigable Small World) algorithm. It's designed for:

- **AI/ML applications** that need fast similarity search
- **Local storage** without external API calls (like SQLite for vectors)
- **High performance** via C implementation with Python bindings

### Key Features

| Feature | Description |
|---------|-------------|
| **Algorithm** | HNSW with M=18, convergent search |
| **Distance Metric** | Cosine similarity (default) |
| **Dimensions** | Auto-detected from first embedding |
| **File Format** | Binary `.hnsw` files |
| **Language** | C core + Python wrapper via CFFI |

### Use Case

```python
# Instead of a hosted vector service:
# vectors are stored, searched and persisted locally.
import hnswdb
db = hnswdb.HNSWDatabase()
db.start()
db.store(my_embeddings)
results = db.find(query_embedding, k=5)
db.save("my_vectors.hnsw")
```

---

## 2. Architecture

### Directory Structure

```
hnswdb/
├── hnswdb/                     # Main package
│   ├── __init__.py             # Package exports
│   ├── database.py             # Python wrapper class
│   ├── build_cffi.py           # CFFI build script (generates _hnsw_cffi.c)
│   └── core/                   # C implementation
│       ├── platform.h          # Cross-platform compatibility
│       ├── hnsw_types.h        # Type definitions & constants
│       ├── priority_queue.c/h  # Binary heap data structure
│       ├── distance.c/h        # Distance/similarity functions
│       ├── hnsw_io.c/h         # File persistence
│       └── hnsw.c/h            # Core HNSW algorithm
├── tests/                      # Test suite
│   ├── test_hnswdb.py          # Core API tests
│   └── test_comprehensive.py   # Edge cases + brute-force correctness checks
├── examples/                   # Usage examples
│   └── law_search.py
├── setup.py                    # Registers the CFFI extension
├── pyproject.toml              # Package metadata & build config
├── README.md                   # Quick start guide
├── LICENSE                     # MIT License
└── MANIFEST.in                 # Additional files for sdist
```

### Component Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                     Python Layer                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              database.py (HNSWDatabase)              │   │
│  │  start() | store() | find() | save() | open() | ... │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 CFFI Bindings                        │   │
│  │            _hnsw_cffi (auto-generated)               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       C Layer                               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────┐  │
│  │  hnsw.c   │  │ distance.c│  │ hnsw_io.c │  │priority │  │
│  │  (core)   │  │ (metrics) │  │  (file)   │  │_queue.c │  │
│  └───────────┘  └───────────┘  └───────────┘  └─────────┘  │
│        │              │              │              │       │
│        └──────────────┴──────────────┴──────────────┘       │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         hnsw.h (public API, error codes)             │   │
│  │         hnsw_types.h (internal types)                │   │
│  │         platform.h (OS abstraction)                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

The code follows SOLID principles:

| Principle | Implementation |
|-----------|----------------|
| **S**ingle Responsibility | Each C file handles one concern |
| **O**pen/Closed | Distance functions extensible without modification |
| **L**iskov Substitution | All distance functions share same signature |
| **I**nterface Segregation | Minimal public API in hnsw.h |
| **D**ependency Inversion | Core depends on abstractions (function pointers) |

---

## 3. HNSW Algorithm Explained

### Concept

HNSW creates a **multi-layer graph** where:
- Higher layers have sparse long-range connections (fast traversal)
- Lower layers have dense local connections (precise search)
- Search starts at top, greedy-descends to bottom

```
Level 3:  [A]─────────────────────────[B]
           │                           │
Level 2:  [A]────────[C]──────────────[B]
           │          │                │
Level 1:  [A]───[D]──[C]───[E]───[F]──[B]
           │    │    │     │     │    │
Level 0:  [A]─[D]─[G]─[C]─[H]─[E]─[F]─[B]─[I]  (all nodes)
```

### Algorithm Parameters (Fixed)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **M** | 18 | Max connections per node (per layer) |
| **M_MAX** | 36 | Max connections at layer 0 (denser) |
| **Level Multiplier** | 1/ln(18) ≈ 0.346 | Probability factor for level assignment |

### Why No ef Parameter?

Traditional HNSW uses `ef` (expansion factor) to limit search breadth. We use **convergent search** instead:

```
Traditional:  Stop when candidates.size >= ef
Ours:         Stop when current.distance > worst_of_k.distance
```

**Result:** No `ef` tuning knob. The search keeps expanding while any candidate can still improve the current top-k, then stops. Like any graph-based ANN index, results are still *approximate*: recall depends on graph connectivity. The test suite checks results against brute-force k-NN.

### Insertion Algorithm

```
INSERT(vector):
    1. Assign random level L using exponential distribution
    2. Start at entry_point (top of graph)

    3. For each layer from max_level down to L+1:
       - Greedy search: find single closest node
       - This becomes entry point for next layer

    4. For each layer from min(L, max_level) down to 0:
       - Search for M candidate neighbors
       - Select M closest
       - Create bidirectional edges

    5. If L > max_level:
       - New node becomes entry_point
       - Update max_level = L
```

### Search Algorithm

```
SEARCH(query, k):
    1. Start at entry_point

    2. For each layer from max_level down to 1:
       - Greedy search: find single closest node
       - This becomes entry point for next layer

    3. At layer 0:
       - Convergent search for k neighbors
       - Keep searching until no unvisited node
         is closer than current k-th best

    4. Return k closest nodes
```

### Distance Calculation

**Cosine Distance** (default):
```
cosine_distance(a, b) = 1 - (a · b) / (||a|| × ||b||)

Range: [0, 2]
  - 0 = identical direction
  - 1 = perpendicular
  - 2 = opposite direction
```

---

## 4. C Core Implementation

### 4.1 platform.h - Cross-Platform Layer

**Purpose:** Abstract OS differences (Windows vs Unix)

```c
// Thread-safe random (Windows lacks rand_r)
static HNSW_INLINE int hnsw_rand_r(unsigned int* seed) {
#ifdef _WIN32
    *seed = *seed * 1103515245 + 12345;
    return (int)((*seed / 65536) % 32768);
#else
    return rand_r(seed);
#endif
}

// Export macros
#ifdef _WIN32
    #define HNSW_API __declspec(dllexport)
#else
    #define HNSW_API __attribute__((visibility("default")))
#endif
```

### 4.2 hnsw_types.h - Type Definitions

**Constants:**
```c
#define HNSW_M                 18      // Connections per layer
#define HNSW_M_MAX             36      // Connections at layer 0
#define HNSW_LEVEL_MULTIPLIER  0.346f  // 1/ln(18)
#define HNSW_INITIAL_CAPACITY  1024    // Initial node array size
```

**Node Structure:**
```c
typedef struct {
    float*    vector;           // The embedding data
    int32_t*  neighbors;        // Flat array of neighbor IDs
    int32_t*  neighbor_counts;  // Count per level
    int32_t   max_level;        // Highest level this node exists at
    int32_t   id;               // Node identifier (0-indexed)
} HNSWNode;
```

**Index Structure:**
```c
struct HNSWIndex {
    HNSWNode*     nodes;            // Array of all nodes
    int32_t       node_count;       // Current count
    int32_t       node_capacity;    // Allocated capacity
    int32_t       dim;              // Vector dimension (0 = auto)
    int32_t       entry_point;      // Top-level entry node
    int32_t       max_level;        // Current max level
    HNSWDistanceFunc distance_func; // Distance function pointer
    unsigned int  random_seed;      // For level generation
};
```

### 4.3 priority_queue.c/h - Binary Heap

**Purpose:** Manage search candidates efficiently

```c
// Create min-heap (for candidates) or max-heap (for results)
PriorityQueue* pq_create(int32_t capacity, PQHeapType type);

// O(log n) operations
bool pq_push(PriorityQueue* pq, int32_t id, float distance);
PQNode pq_pop(PriorityQueue* pq);   // Remove top
PQNode pq_top(const PriorityQueue* pq);  // Peek top

// O(1) queries
bool pq_is_empty(const PriorityQueue* pq);
int32_t pq_size(const PriorityQueue* pq);
```

**Heap Property:**
- Min-heap: Parent ≤ children (closest candidate at top)
- Max-heap: Parent ≥ children (farthest result at top for pruning)

### 4.4 distance.c/h - Distance Functions

**Cosine Distance:**
```c
float distance_cosine(const float* a, const float* b, int32_t dim) {
    float dot = 0, norm_a = 0, norm_b = 0;

    for (int32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < 1e-10f) return 1.0f;  // Zero vector

    float similarity = dot / denom;
    return 1.0f - similarity;  // Distance = 1 - similarity
}
```

**Factory Pattern:**
```c
HNSWDistanceFunc distance_get_function(HNSWDistanceType type) {
    switch (type) {
        case HNSW_DISTANCE_COSINE: return distance_cosine;
        case HNSW_DISTANCE_EUCLIDEAN: return distance_euclidean;
        default: return distance_cosine;
    }
}
```

### 4.5 hnsw.c - Core Algorithm

**Visited Set (Bitmap):**
```c
typedef struct {
    uint32_t* bits;   // 1 bit per node
    int32_t capacity; // Number of uint32_t words
} VisitedSet;

static inline void visited_set(VisitedSet* vs, int32_t id) {
    vs->bits[id / 32] |= (1u << (id % 32));
}

static inline int visited_get(const VisitedSet* vs, int32_t id) {
    return (vs->bits[id / 32] & (1u << (id % 32))) != 0;
}
```

**Level Generation:**
```c
static int32_t generate_random_level(HNSWIndex* index) {
    float r = (float)hnsw_rand_r(&index->random_seed) / RAND_MAX;
    if (r < 1e-9f) r = 1e-9f;  // Avoid log(0)
    return (int32_t)(-logf(r) * HNSW_LEVEL_MULTIPLIER);
}
```

**Convergent Layer Search:**
```c
static PriorityQueue* search_layer(HNSWIndex* index, const float* query,
                                    int32_t entry_id, int32_t level, int32_t k) {
    VisitedSet* visited = visited_create(index->node_count);
    PriorityQueue* candidates = pq_create(64, PQ_MIN_HEAP);  // Best first
    PriorityQueue* results = pq_create(64, PQ_MAX_HEAP);     // Worst on top

    // Initialize with entry point
    float entry_dist = distance(query, nodes[entry_id].vector, dim);
    pq_push(candidates, entry_id, entry_dist);
    pq_push(results, entry_id, entry_dist);
    visited_set(visited, entry_id);

    while (!pq_is_empty(candidates)) {
        PQNode current = pq_pop(candidates);

        // Stop if current can't improve results
        if (pq_size(results) >= k && current.distance > pq_top(results).distance) {
            break;  // CONVERGENCE
        }

        // Explore all neighbors
        for each neighbor_id in current.neighbors[level]:
            if (!visited_get(visited, neighbor_id)):
                visited_set(visited, neighbor_id);
                float dist = distance(query, nodes[neighbor_id].vector, dim);

                if (pq_size(results) < k || dist < pq_top(results).distance):
                    pq_push(candidates, neighbor_id, dist);
                    pq_push(results, neighbor_id, dist);
                    if (pq_size(results) > k):
                        pq_pop(results);  // Remove worst
    }

    return results;
}
```

### 4.6 hnsw_io.c/h - File Persistence

**File Header:**
```c
typedef struct {
    uint32_t magic;         // 0x57534E48 = "HNSW"
    uint32_t version;       // Format version (currently 2)
    int32_t  dim;           // Vector dimension
    int32_t  node_count;    // Number of vectors
    int32_t  max_level;     // Maximum graph level
    int32_t  entry_point;   // Entry node ID
    int32_t  distance_type; // Distance metric enum
    int32_t  reserved[5];   // Future expansion
} HNSWFileHeader;
```

**Save Process:**
```
1. Write header (48 bytes)
2. For each node:
   a. Write vector (dim × 4 bytes)
   b. Write max_level (4 bytes)
   c. For each level 0..max_level:
      - Write neighbor_count (4 bytes)
      - Write neighbor_ids (count × 4 bytes)
```

---

## 5. Python API Reference

### HNSWDatabase Class

```python
class HNSWDatabase:
    """
    Local vector database using HNSW algorithm.

    Lifecycle:
        1. __init__()  - Create instance
        2. start()     - Initialize structures
        3. store()/open() - Add or load data
        4. find()      - Search
        5. save()      - Persist
        6. close()     - Cleanup
    """
```

### Methods

#### `__init__()`
Create a new database instance (not yet initialized).

```python
db = hnswdb.HNSWDatabase()
```

#### `start() -> HNSWDatabase`
Initialize in-memory structures. Must be called before other operations.

```python
db.start()
```

#### `store(embeddings: List[List[float]]) -> HNSWDatabase`
Store embeddings in a new index. Dimension is auto-detected.

```python
embeddings = [
    [0.1, 0.2, 0.3],  # 3D vector
    [0.4, 0.5, 0.6],
]
db.store(embeddings)
```

**Raises:**
- `ValueError` if embeddings is empty
- `DimensionMismatchError` if vectors have different dimensions

#### `find(target: List[float], k: int = 5) -> List[List[float]]`
Find k nearest neighbor embeddings.

```python
query = [0.15, 0.25, 0.35]
neighbors = db.find(query, k=3)
# Returns: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...]
```

#### `find_with_distances(target: List[float], k: int = 5) -> List[Tuple[List[float], float]]`
Find neighbors with their distance scores.

```python
results = db.find_with_distances(query, k=3)
for embedding, distance in results:
    print(f"Distance: {distance:.4f}")
# Note: distance is cosine distance (0 = identical, 2 = opposite)
```

#### `find_ids(target: List[float], k: int = 5) -> List[Tuple[int, float]]`
Find neighbor IDs and distances (faster, no vector data).

```python
results = db.find_ids(query, k=3)
for idx, distance in results:
    print(f"ID: {idx}, Distance: {distance:.4f}")
```

#### `save(path: str = None) -> str`
Save index to disk. Returns actual path used.

```python
path = db.save("vectors.hnsw")
# If file exists: returns "vectors1.hnsw", "vectors2.hnsw", etc.
```

#### `open(path: str) -> HNSWDatabase`
Load existing index from disk.

```python
db.start()
db.open("vectors.hnsw")
```

#### `append(embeddings: List[List[float]]) -> HNSWDatabase`
Add embeddings to existing index. Auto-saves if loaded from file.

```python
db.open("vectors.hnsw")
db.append([[0.7, 0.8, 0.9]])  # Adds and saves
```

#### `close()`
Free resources and cleanup.

```python
db.close()
```

#### `clear()`
**Permanently delete** the saved `.hnsw` file (if any) and free all in-memory data. Call `start()` again before reusing the instance.

```python
db.clear()   # file on disk is removed
db.start()   # ready for a fresh index
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `count` | int | Number of stored embeddings |
| `dimension` | int | Vector dimension (None if empty) |
| `is_saved` | bool | Whether current state is saved |
| `save_path` | str | Path to save file (None if not saved) |

### Context Manager

```python
with hnswdb.HNSWDatabase() as db:
    db.store(embeddings)
    results = db.find(query)
# Automatically calls close()
```

### Exceptions

| Exception | Description |
|-----------|-------------|
| `HNSWError` | Base exception for all errors |
| `DimensionMismatchError` | Vector dimensions don't match |
| `NotInitializedError` | Operations on unstarted database |
| `FileIOError` | File read/write failures |

---

## 6. File Format Specification

### Overview

Binary format with `.hnsw` extension.

### Structure

```
┌────────────────────────────────────────┐
│            Header (48 bytes)           │
├────────────────────────────────────────┤
│   magic     │ "HNSW" = 0x57534E48     │ 4 bytes
│   version   │ 2                        │ 4 bytes
│   dim       │ Vector dimension         │ 4 bytes
│   count     │ Number of nodes          │ 4 bytes
│   max_level │ Maximum graph level      │ 4 bytes
│   entry_pt  │ Entry point node ID      │ 4 bytes
│   dist_type │ Distance metric          │ 4 bytes
│   reserved  │ (future use)             │ 20 bytes
├────────────────────────────────────────┤
│              Nodes                     │
├────────────────────────────────────────┤
│ Node 0:                                │
│   vector    │ dim × float32            │ dim×4 bytes
│   max_level │ int32                    │ 4 bytes
│   Level 0:                             │
│     count   │ int32                    │ 4 bytes
│     ids     │ count × int32            │ count×4 bytes
│   Level 1:  (if max_level >= 1)        │
│     count   │ int32                    │ 4 bytes
│     ids     │ count × int32            │ count×4 bytes
│   ...                                  │
├────────────────────────────────────────┤
│ Node 1: ...                            │
│ Node 2: ...                            │
│ ...                                    │
└────────────────────────────────────────┘
```

### Version History

| Version | Changes |
|---------|---------|
| 1 | Initial format |
| 2 | Added reserved fields, distance_type |

### File Size Estimation

```
Header: 48 bytes
Per node: dim×4 + 4 + Σ(4 + neighbors×4) bytes

Example (1M vectors, 384-dim, avg 25 neighbors):
  1,000,000 × (384×4 + 4 + 30) + 48 ≈ 1.5 GB
```

---

## 7. Build System

### CFFI Build Process

The `build_cffi.py` script:

1. Declares the public C interface (`ffi.cdef`), mirroring `hnsw.h`
2. Registers each C module (`priority_queue.c`, `distance.c`, `hnsw_io.c`, `hnsw.c`) as a separate translation unit
3. Generates the CFFI glue (`_hnsw_cffi.c`) and compiles everything into one extension module

```python
# build_cffi.py structure
ffi = FFI()
ffi.cdef("...")  # Declare C interface
ffi.set_source("hnswdb._hnsw_cffi", '#include "hnsw.h"', sources=[...])
ffi.compile()
```

### Building Manually

From the repository root:

```bash
pip install cffi
python hnswdb/build_cffi.py   # writes the compiled extension into hnswdb/
```

### Installing from Source

```bash
pip install .
```

This triggers the CFFI build via `setup.py`.

### Requirements

- Python 3.9+
- C compiler:
  - Windows: MSVC (Visual Studio Build Tools)
  - Linux: GCC
  - macOS: Clang (Xcode Command Line Tools)
- cffi package

---

## 8. Configuration & Constants

### Compile-Time Constants (hnsw_types.h)

| Constant | Value | Meaning |
|----------|-------|---------|
| `HNSW_M` | 18 | Connections per node (layers 1+) |
| `HNSW_M_MAX` | 36 | Connections at layer 0 |
| `HNSW_LEVEL_MULTIPLIER` | 0.346 | 1/ln(M) for level probability |
| `HNSW_INITIAL_CAPACITY` | 1024 | Initial node array size |
| `HNSW_ALIGNMENT` | 32 | Memory alignment (SIMD-ready) |

### File Format Constants (hnsw_io.h)

| Constant | Value | Meaning |
|----------|-------|---------|
| `HNSW_FILE_MAGIC` | 0x57534E48 | "HNSW" identifier |
| `HNSW_FILE_VERSION` | 2 | Current format version |
| `HNSW_FILE_MIN_VERSION` | 1 | Oldest readable version |

### Modifying Parameters

To change M:

1. Edit `HNSW_M` in `hnsw_types.h`
2. Update `HNSW_LEVEL_MULTIPLIER` to `1/ln(M)`
3. Rebuild: `python hnswdb/build_cffi.py`

**Tradeoffs:**
- Higher M → Better search quality, more memory, slower insert
- Lower M → Faster insert, less memory, worse search quality

---

## 9. Error Handling

### Error Codes (C, defined in hnsw.h)

```c
#define HNSW_OK                     0  // Success
#define HNSW_ERR_MEMORY            -1  // malloc failed
#define HNSW_ERR_INVALID_PARAM     -2  // NULL pointer, negative k
#define HNSW_ERR_NOT_INITIALIZED   -3  // Index not created
#define HNSW_ERR_FILE_IO           -4  // fopen/fread/fwrite failed
#define HNSW_ERR_DIMENSION_MISMATCH -5 // Query dim != index dim
#define HNSW_ERR_EMPTY_INDEX       -6  // No vectors stored
#define HNSW_ERR_INVALID_FILE      -7  // Bad magic number
#define HNSW_ERR_VERSION_MISMATCH  -8  // Unsupported version
```

### Python Exception Mapping

| C Error Code | Python Exception |
|--------------|------------------|
| HNSW_ERR_MEMORY | `HNSWError` |
| HNSW_ERR_DIMENSION_MISMATCH | `DimensionMismatchError` |
| HNSW_ERR_FILE_IO | `FileIOError` |
| HNSW_ERR_EMPTY_INDEX | Returns empty list |

---

## 10. Performance Characteristics

### Time Complexity

| Operation | Average | Worst |
|-----------|---------|-------|
| Insert one vector | O(log n) | O(n) |
| Search k neighbors | O(log n) | O(n) |
| Save to file | O(n) | O(n) |
| Load from file | O(n) | O(n) |

### Space Complexity

```
Memory per node ≈ dim×4 + max_level×M×4 bytes

For dim=384, avg_level=2, M=18:
  ≈ 384×4 + 2×18×4 = 1536 + 144 = 1680 bytes/vector

1 million vectors ≈ 1.6 GB RAM
```

### Benchmarks

Formal benchmarks have not been published yet. Search cost grows roughly logarithmically with index size. Insert cost is dominated by the layer-0 neighbor search and the O(n²) neighbor sort noted under *Known Issues*. Performance depends on hardware, dimension and data distribution.

---

## 11. Limitations & Future Work

### Current Limitations

1. **Not thread-safe** - Don't share index across threads
2. **No deletion** - Can't remove individual vectors
3. **No incremental save** - Full save each time
4. **Scalar distance** - No SIMD optimization yet
5. **Memory-resident** - Full index must fit in RAM

### Planned Improvements

1. **SIMD distance functions** - AVX/SSE for 4-8x speedup
2. **Batch search API** - Search multiple queries at once
3. **Delete support** - Mark nodes as deleted, compact on save
4. **Memory mapping** - Load large indices from disk
5. **Thread-safe search** - Parallel read-only queries

### Known Issues

1. **Over-allocation of neighbor storage** - Uses M_MAX slots for all levels
2. **O(n²) sort in select_neighbors** - Could use quickselect for O(n)
3. **Full vector retrieval in find()** - Should add get_vector_by_id()

---

## Quick Reference

### Typical Workflow

```python
import hnswdb

# Create and populate
db = hnswdb.HNSWDatabase()
db.start()
db.store(my_embeddings)  # Any dimension
db.save("index.hnsw")
db.close()

# Later: load and search
db = hnswdb.HNSWDatabase()
db.start()
db.open("index.hnsw")
results = db.find(query, k=5)
db.close()
```

### Distance Interpretation

Cosine distance values:
- **0.0** - Identical vectors
- **0.1-0.3** - Very similar
- **0.3-0.5** - Somewhat similar
- **0.5-1.0** - Dissimilar
- **1.0-2.0** - Opposite direction

### Choosing k

- **k=1** - Single best match
- **k=5-10** - Typical for recommendations
- **k=100+** - For reranking or analysis

---

*Author: [Hariharan Thangaraj](https://github.com/Hariharan-1810)*
