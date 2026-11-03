"""
HNSWDB - Local HNSW Vector Database

A high-performance local vector database using the HNSW algorithm.
Store and search embeddings directly on your system without external API calls.

Installation:
    pip install git+https://github.com/Hariharan-1810/hnswdb.git

Usage:
    import hnswdb

    # Create and use database
    db = hnswdb.HNSWDatabase()
    db.start()

    # Store embeddings
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    db.store(embeddings)

    # Find nearest neighbors
    query = [0.15, 0.25, 0.35]
    neighbors = db.find(query, k=2)

    # Save to disk
    db.save("my_vectors.hnsw")

    # Later, open saved database
    db2 = hnswdb.HNSWDatabase()
    db2.start()
    db2.open("my_vectors.hnsw")

    # Append more vectors
    db2.append([[1.0, 1.1, 1.2]])

    # Clean up
    db2.close()

Technical Details:
    - Algorithm: HNSW (Hierarchical Navigable Small World)
    - M = 18 (bi-directional connections per node, 36 at layer 0)
    - Convergent best-first search (no fixed ef cut-off)
    - Distance: Cosine distance (1 - cosine similarity)
    - File extension: .hnsw
"""

__version__ = "0.1.0"
__author__ = "Hariharan Thangaraj"

from .database import (
    HNSWDatabase,
    HNSWError,
    DimensionMismatchError,
    NotInitializedError,
    FileIOError,
)

__all__ = [
    "HNSWDatabase",
    "HNSWError",
    "DimensionMismatchError",
    "NotInitializedError",
    "FileIOError",
    "__version__",
]
