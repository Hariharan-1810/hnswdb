# HNSWDB: a local vector database in C with Python bindings

[![CI](https://github.com/Hariharan-1810/hnswdb/actions/workflows/ci.yml/badge.svg)](https://github.com/Hariharan-1810/hnswdb/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

HNSWDB is an embedded vector database built from scratch. It implements the
**HNSW (Hierarchical Navigable Small World)** graph algorithm
([Malkov & Yashunin, 2016](https://arxiv.org/abs/1603.09320)) in plain C and exposes it
to Python through CFFI. You can store embeddings, run fast nearest-neighbour search and
persist the index to a single `.hnsw` file, all without a server or network calls.

Think of it as *SQLite for embeddings*: `import`, store, search, save.

```python
import hnswdb

with hnswdb.HNSWDatabase() as db:
    db.store(embeddings)                  # any dimension, auto-detected
    ids = db.find_ids(query, k=5)         # [(id, cosine_distance), ...]
    db.save("knowledge.hnsw")             # one portable binary file
```

---

## Highlights

- **HNSW implemented from scratch in C99:** a multi-layer proximity graph,
  probabilistic level assignment, bidirectional edges with worst-neighbour replacement,
  and a bitmap visited-set.
- **Custom binary file format** with a magic number, a version and reserved header
  space for forward compatibility (spec in [DOCUMENTATION.md](DOCUMENTATION.md#6-file-format-specification)).
- **Modular C core:** graph, priority queue, distance metrics and I/O live in separate
  translation units behind a small opaque-handle public API (`hnsw.h`).
- **Pluggable distance metrics** through function pointers (cosine by default; Euclidean
  and inner product are implemented).
- **Pythonic wrapper** with typed exceptions, a context manager, method chaining and
  automatic save on `append()`.
- **Tested against brute-force k-NN**, with 50+ tests covering correctness, persistence
  and edge cases (zero vectors, 1536-dim vectors, duplicates and more).
- **Cross-platform CI** on Linux, macOS and Windows.

## How it works

```
Layer 2:   A ─────────────────── B           sparse, long-range links
           │                     │
Layer 1:   A ───── C ─────── E ─ B
           │       │         │   │
Layer 0:   A ─ D ─ C ─ F ─ G ─ E ─ B ─ H     every vector, dense local links
```

1. **Insert:** each vector draws a random top layer from an exponential distribution
   (multiplier `1/ln(M)`). Starting at the entry point, the algorithm greedily descends
   to that layer, then connects the new node to its nearest neighbours on every layer
   below. That is up to `M = 18` links per layer, or `2M = 36` on the dense layer 0.
2. **Search:** greedy descent through the upper layers finds a good entry point. A
   best-first search on layer 0 then keeps expanding while any candidate can still
   improve the current top-k, so there is no `ef` constant to tune.
3. **Persist:** the full graph (vectors and adjacency lists per layer) is serialised to
   a compact little-endian binary file and loaded back without rebuilding.

## Architecture

```
hnswdb/
├── hnswdb/
│   ├── database.py        # Python API: HNSWDatabase, typed exceptions
│   ├── build_cffi.py      # CFFI build: compiles the C core into an extension
│   └── core/              # C99 implementation
│       ├── hnsw.c/.h      # graph construction + search (public API)
│       ├── priority_queue # binary min/max heap for candidates/results
│       ├── distance       # cosine / L2 / inner-product (strategy pattern)
│       ├── hnsw_io        # binary save/load + header validation
│       ├── hnsw_types.h   # internal structs & tuning constants
│       └── platform.h     # Windows / POSIX portability shims
├── tests/                 # pytest suite incl. brute-force correctness checks
└── examples/              # end-to-end example
```

## Installation

You need Python 3.10+ and a C compiler (GCC/Clang, or MSVC Build Tools on Windows).

```bash
pip install git+https://github.com/Hariharan-1810/hnswdb.git
```

From source, for development:

```bash
git clone https://github.com/Hariharan-1810/hnswdb.git
cd hnswdb
pip install cffi pytest
python hnswdb/build_cffi.py     # compile the C extension in place
pytest tests -v
```

## Quick start

```python
import hnswdb

db = hnswdb.HNSWDatabase()
db.start()

db.store([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.2, 0.3, 0.4, 0.5],
])

print(db.find_ids([0.15, 0.25, 0.35, 0.45], k=2))   # [(id, cosine_distance), ...] closest first

path = db.save("my_vectors.hnsw")
db.close()

# later
db = hnswdb.HNSWDatabase().start()
db.open(path)
db.append([[0.9, 0.8, 0.7, 0.6]])   # auto-saves, because the index came from a file
db.close()
```

## API at a glance

| Method | Description |
|---|---|
| `start()` | Initialise an empty in-memory index |
| `store(embeddings)` | Insert vectors (dimension auto-detected on first insert) |
| `find(query, k=5)` | Top-k nearest vectors |
| `find_with_distances(query, k=5)` | `[(vector, distance), ...]` |
| `find_ids(query, k=5)` | `[(id, distance), ...]`, the fastest option |
| `save(path=None)` | Persist to `.hnsw`; never overwrites (`x.hnsw` becomes `x1.hnsw`) |
| `open(path)` | Load a saved index |
| `append(embeddings)` | Insert into a loaded index and auto-save |
| `close()` | Free memory; saved files are kept |
| `clear()` | Free memory **and delete** the saved file |
| `count`, `dimension`, `is_saved`, `save_path` | Properties |

Exceptions: `HNSWError` (base), `DimensionMismatchError`, `NotInitializedError`,
`FileIOError`. See [DOCUMENTATION.md](DOCUMENTATION.md) for the full reference.

## Example: semantic search / RAG retrieval

```python
import os
import hnswdb

class LocalVectorStore:
    def __init__(self, path="knowledge.hnsw"):
        self.path = path
        self.db = hnswdb.HNSWDatabase().start()
        if os.path.exists(path):
            self.db.open(path)

    def add(self, embeddings):
        if self.db.count:
            self.db.append(embeddings)            # auto-saves
        else:
            self.db.store(embeddings)
            self.path = self.db.save(self.path)

    def search(self, query_embedding, k=5):
        return self.db.find_ids(query_embedding, k)   # map ids -> your documents
```

A runnable end-to-end demo is in [`examples/law_search.py`](examples/law_search.py).

## Design decisions

| Decision | Why |
|---|---|
| C core and a CFFI wrapper | Tight loops (distance and graph traversal) run in native code; CFFI keeps the binding thin and needs no Python C-API boilerplate |
| Opaque `HNSWIndex*` handle | The internal layout can change without breaking the public API or ABI |
| Function-pointer distance metrics | New metrics can be added without touching graph code |
| Bitmap visited-set | 1 bit per node, so traversal state stays cheap and cache-friendly |
| Versioned binary format | Fast load (no rebuild) with a path for evolving the format |

## Roadmap

- [ ] SIMD (AVX2/NEON) distance kernels
- [ ] Neighbour-selection diversity heuristic (from the HNSW paper) and a tunable `ef`
- [ ] Faster neighbour sort at insert time (currently O(n²))
- [ ] Deletion via tombstones and compaction on save
- [ ] Atomic saves (temp file and rename), and memory-mapped loading
- [ ] Batch search and thread-safe concurrent reads
- [ ] Published benchmarks against brute force (recall@k and QPS)

## License

MIT © [Hariharan Thangaraj](https://github.com/Hariharan-1810)
