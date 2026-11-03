"""
HNSW Vector Database - Python Wrapper

A high-performance local vector database using HNSW algorithm.
Store and search embeddings directly on your system without external API calls.

Usage:
    import hnswdb

    db = hnswdb.HNSWDatabase()
    db.start()
    db.store(embeddings)  # Create new DB with embeddings
    results = db.find(query_embedding, k=5)  # Find 5 nearest neighbors
    db.save("mydata.hnsw")  # Save permanently
    db.close()
"""

import os
import re
import atexit
import tempfile
from typing import List, Optional, Tuple, Union
from pathlib import Path

try:
    from ._hnsw_cffi import ffi, lib
except ImportError:
    # For development/build time
    ffi = None
    lib = None


class HNSWError(Exception):
    """Base exception for HNSW database errors"""
    pass


class DimensionMismatchError(HNSWError):
    """Raised when embedding dimensions don't match"""
    pass


class NotInitializedError(HNSWError):
    """Raised when operations are attempted on uninitialized database"""
    pass


class FileIOError(HNSWError):
    """Raised when file operations fail"""
    pass


class HNSWDatabase:
    """
    HNSW Vector Database

    A local vector database that stores and retrieves embeddings using the
    Hierarchical Navigable Small World (HNSW) algorithm for fast approximate
    nearest neighbor search.

    Parameters:
        M (int): Fixed at 18 - maximum bi-directional connections per node

    Example:
        >>> db = HNSWDatabase()
        >>> db.start()
        >>> db.store([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> results = db.find([0.1, 0.2, 0.3], k=1)
        >>> db.save("vectors.hnsw")
        >>> db.close()
    """

    DEFAULT_EXTENSION = ".hnsw"

    def __init__(self):
        """Initialize HNSWDatabase instance (not started yet)"""
        self._index = None
        self._is_started = False
        self._is_saved = False
        self._save_path: Optional[str] = None
        self._temp_path: Optional[str] = None
        self._dimension: Optional[int] = None

        # Register cleanup on exit
        atexit.register(self._cleanup_on_exit)

    def _check_lib(self):
        """Check if the C library is available"""
        if lib is None:
            raise HNSWError(
                "HNSW C library not compiled. Run 'python hnswdb/build_cffi.py' first."
            )

    def _check_started(self):
        """Ensure database is started"""
        if not self._is_started or self._index is None:
            raise NotInitializedError("Database not started. Call start() first.")

    def _check_dimensions(self, embeddings: List[List[float]]) -> int:
        """Validate and return embedding dimension"""
        if not embeddings:
            raise ValueError("Embeddings list cannot be empty")

        dim = len(embeddings[0])
        if dim == 0:
            raise ValueError("Embedding dimension cannot be zero")

        for i, emb in enumerate(embeddings):
            if len(emb) != dim:
                raise DimensionMismatchError(
                    f"Embedding at index {i} has dimension {len(emb)}, expected {dim}"
                )

        if self._dimension is not None and self._dimension != dim:
            raise DimensionMismatchError(
                f"New embeddings have dimension {dim}, but database has dimension {self._dimension}"
            )

        return dim

    def _embeddings_to_cffi(self, embeddings: List[List[float]]) -> Tuple:
        """Convert Python list of embeddings to CFFI array"""
        n = len(embeddings)
        dim = len(embeddings[0])

        # Flatten embeddings
        flat = []
        for emb in embeddings:
            flat.extend(emb)

        # Create CFFI array
        arr = ffi.new(f"float[{n * dim}]", flat)
        return arr, n, dim

    def _cffi_to_embeddings(self, arr, n: int, dim: int) -> List[List[float]]:
        """Convert CFFI array back to Python list of embeddings"""
        result = []
        for i in range(n):
            emb = [arr[i * dim + j] for j in range(dim)]
            result.append(emb)
        return result

    def _generate_filename(self, base_path: str) -> str:
        """
        Generate unique filename if file already exists.
        sample.hnsw -> sample1.hnsw -> sample2.hnsw, etc.
        """
        path = Path(base_path)

        # Ensure correct extension
        if path.suffix != self.DEFAULT_EXTENSION:
            path = path.with_suffix(self.DEFAULT_EXTENSION)

        if not path.exists():
            return str(path)

        # Extract base name without extension
        stem = path.stem
        parent = path.parent

        # Remove existing number suffix if present
        match = re.match(r'^(.+?)(\d+)$', stem)
        if match:
            base_stem = match.group(1)
            start_num = int(match.group(2)) + 1
        else:
            base_stem = stem
            start_num = 1

        # Find next available number
        counter = start_num
        while True:
            new_path = parent / f"{base_stem}{counter}{self.DEFAULT_EXTENSION}"
            if not new_path.exists():
                return str(new_path)
            counter += 1

    def _cleanup_on_exit(self):
        """Cleanup handler called on program exit"""
        if self._is_started and self._index is not None:
            # If not saved, delete temp data
            if not self._is_saved and self._temp_path and os.path.exists(self._temp_path):
                try:
                    os.remove(self._temp_path)
                except:
                    pass

            # Free memory
            try:
                lib.hnsw_destroy(self._index)
            except:
                pass

    # ========== Public API ==========

    def start(self) -> 'HNSWDatabase':
        """
        Initialize the database and prepare all in-memory HNSW structures.

        Returns:
            self: For method chaining

        Example:
            >>> db = HNSWDatabase()
            >>> db.start()
        """
        self._check_lib()

        if self._is_started and self._index is not None:
            # Already started, just return
            return self

        # Create new index (dimension 0 = auto-detect)
        self._index = lib.hnsw_create(0)
        if self._index == ffi.NULL:
            raise HNSWError("Failed to create HNSW index")

        self._is_started = True
        self._is_saved = False
        self._save_path = None
        self._dimension = None

        return self

    def store(self, embeddings: List[List[float]]) -> 'HNSWDatabase':
        """
        Store embeddings in a new HNSW index (volatile until saved).

        This creates a new database. If you want to add to an existing saved
        database, use open() + append() instead.

        Parameters:
            embeddings: List of vectors, where each vector is a list of floats.
                       Dimension is auto-detected from the first embedding.

        Returns:
            self: For method chaining

        Example:
            >>> db.start()
            >>> db.store([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        """
        self._check_started()

        dim = self._check_dimensions(embeddings)
        arr, n, dim = self._embeddings_to_cffi(embeddings)

        result = lib.hnsw_add(self._index, arr, n, dim)

        if result != 0:  # HNSW_OK
            if result == -5:  # HNSW_ERR_DIMENSION_MISMATCH
                raise DimensionMismatchError("Dimension mismatch with existing data")
            elif result == -1:  # HNSW_ERR_MEMORY
                raise HNSWError("Memory allocation failed")
            else:
                raise HNSWError(f"Failed to store embeddings (error code: {result})")

        self._dimension = dim
        self._is_saved = False

        return self

    def find(self, target_embedding: List[float], k: int = 5) -> List[List[float]]:
        """
        Retrieve the top-k nearest embeddings from the current HNSW index.

        Parameters:
            target_embedding: Query vector to find neighbors for
            k: Number of nearest neighbors to return (default: 5)

        Returns:
            List of k nearest neighbor embeddings (each as list of floats)

        Example:
            >>> neighbors = db.find([0.1, 0.2, 0.3], k=3)
            >>> print(len(neighbors))  # 3
        """
        self._check_started()

        if not target_embedding:
            raise ValueError("Target embedding cannot be empty")

        dim = len(target_embedding)
        if self._dimension is not None and dim != self._dimension:
            raise DimensionMismatchError(
                f"Query has dimension {dim}, but database has dimension {self._dimension}"
            )

        count = lib.hnsw_count(self._index)
        if count <= 0:
            return []

        # Prepare query
        query = ffi.new("float[]", target_embedding)

        # Prepare results
        results = ffi.new(f"HNSWResult[{k}]")
        num_results = ffi.new("int32_t*")

        result = lib.hnsw_search(self._index, query, dim, k, results, num_results)

        if result != 0:  # HNSW_OK
            if result == -6:  # HNSW_ERR_EMPTY_INDEX
                return []
            elif result == -5:  # HNSW_ERR_DIMENSION_MISMATCH
                raise DimensionMismatchError("Query dimension doesn't match index")
            else:
                raise HNSWError(f"Search failed (error code: {result})")

        # Get the actual vectors for the results
        n_found = num_results[0]
        if n_found == 0:
            return []

        # Get all vectors from index
        total_count = ffi.new("int32_t*")
        total_dim = ffi.new("int32_t*")
        lib.hnsw_get_vectors(self._index, ffi.NULL, total_count, total_dim)

        all_vectors = ffi.new(f"float[{total_count[0] * total_dim[0]}]")
        lib.hnsw_get_vectors(self._index, all_vectors, total_count, total_dim)

        # Extract neighbor vectors
        neighbors = []
        for i in range(n_found):
            idx = results[i].id
            emb = [all_vectors[idx * total_dim[0] + j] for j in range(total_dim[0])]
            neighbors.append(emb)

        return neighbors

    def find_with_distances(self, target_embedding: List[float], k: int = 5) -> List[Tuple[List[float], float]]:
        """
        Retrieve top-k nearest embeddings with their distances.

        Parameters:
            target_embedding: Query vector
            k: Number of neighbors

        Returns:
            List of (embedding, distance) tuples
        """
        self._check_started()

        if not target_embedding:
            raise ValueError("Target embedding cannot be empty")

        dim = len(target_embedding)
        if self._dimension is not None and dim != self._dimension:
            raise DimensionMismatchError(
                f"Query has dimension {dim}, but database has dimension {self._dimension}"
            )

        count = lib.hnsw_count(self._index)
        if count <= 0:
            return []

        query = ffi.new("float[]", target_embedding)
        results = ffi.new(f"HNSWResult[{k}]")
        num_results = ffi.new("int32_t*")

        result = lib.hnsw_search(self._index, query, dim, k, results, num_results)

        if result != 0:
            if result == -6:
                return []
            raise HNSWError(f"Search failed (error code: {result})")

        n_found = num_results[0]
        if n_found == 0:
            return []

        # Get all vectors
        total_count = ffi.new("int32_t*")
        total_dim = ffi.new("int32_t*")
        lib.hnsw_get_vectors(self._index, ffi.NULL, total_count, total_dim)

        all_vectors = ffi.new(f"float[{total_count[0] * total_dim[0]}]")
        lib.hnsw_get_vectors(self._index, all_vectors, total_count, total_dim)

        # Extract with distances
        neighbors = []
        for i in range(n_found):
            idx = results[i].id
            dist = results[i].distance
            emb = [all_vectors[idx * total_dim[0] + j] for j in range(total_dim[0])]
            neighbors.append((emb, dist))

        return neighbors

    def find_ids(self, target_embedding: List[float], k: int = 5) -> List[Tuple[int, float]]:
        """
        Retrieve top-k nearest neighbor IDs with distances.

        Parameters:
            target_embedding: Query vector
            k: Number of neighbors

        Returns:
            List of (id, distance) tuples
        """
        self._check_started()

        if not target_embedding:
            raise ValueError("Target embedding cannot be empty")

        dim = len(target_embedding)
        count = lib.hnsw_count(self._index)
        if count <= 0:
            return []

        query = ffi.new("float[]", target_embedding)
        results = ffi.new(f"HNSWResult[{k}]")
        num_results = ffi.new("int32_t*")

        result = lib.hnsw_search(self._index, query, dim, k, results, num_results)

        if result != 0:
            if result == -6:
                return []
            raise HNSWError(f"Search failed (error code: {result})")

        return [(results[i].id, results[i].distance) for i in range(num_results[0])]

    def save(self, path: Optional[str] = None) -> str:
        """
        Persist the current HNSW index permanently to disk.

        Parameters:
            path: Optional file path. If not provided, generates a default name.
                  If file exists, appends number (sample.hnsw -> sample1.hnsw)

        Returns:
            str: The actual path where the file was saved

        Example:
            >>> actual_path = db.save("mydata.hnsw")
            >>> print(actual_path)  # "mydata.hnsw" or "mydata1.hnsw" if exists
        """
        self._check_started()

        if lib.hnsw_count(self._index) <= 0:
            raise HNSWError("Cannot save empty index")

        # Determine path
        if path is None:
            path = f"vectors{self.DEFAULT_EXTENSION}"

        # Ensure correct extension
        if not path.endswith(self.DEFAULT_EXTENSION):
            path = path + self.DEFAULT_EXTENSION

        # Generate unique filename if exists
        final_path = self._generate_filename(path)

        result = lib.hnsw_save(self._index, final_path.encode('utf-8'))

        if result != 0:
            if result == -4:  # HNSW_ERR_FILE_IO
                raise FileIOError(f"Failed to write to {final_path}")
            else:
                raise HNSWError(f"Failed to save index (error code: {result})")

        self._is_saved = True
        self._save_path = final_path

        return final_path

    def open(self, path: str) -> 'HNSWDatabase':
        """
        Open an existing saved HNSW index from disk.

        This loads an existing database file for reading and appending.

        Parameters:
            path: Path to saved HNSW file

        Returns:
            self: For method chaining

        Example:
            >>> db = HNSWDatabase()
            >>> db.start()
            >>> db.open("mydata.hnsw")
            >>> results = db.find(query, k=5)
        """
        self._check_lib()

        # Ensure extension
        if not path.endswith(self.DEFAULT_EXTENSION):
            path = path + self.DEFAULT_EXTENSION

        if not os.path.exists(path):
            raise FileNotFoundError(f"HNSW file not found: {path}")

        # If already have an index, destroy it first
        if self._index is not None:
            lib.hnsw_destroy(self._index)

        # Load from file
        self._index = lib.hnsw_load(path.encode('utf-8'))

        if self._index == ffi.NULL:
            raise FileIOError(f"Failed to load HNSW file: {path}")

        self._is_started = True
        self._is_saved = True
        self._save_path = path
        self._dimension = lib.hnsw_dimension(self._index)

        return self

    def append(self, embeddings: List[List[float]]) -> 'HNSWDatabase':
        """
        Add new embeddings to an existing HNSW index.

        The changes are automatically saved if the database was loaded from
        or previously saved to a file.

        Parameters:
            embeddings: List of vectors to append

        Returns:
            self: For method chaining

        Example:
            >>> db.open("mydata.hnsw")
            >>> db.append([[0.7, 0.8, 0.9]])  # Adds and auto-saves
        """
        self._check_started()

        dim = self._check_dimensions(embeddings)
        arr, n, dim = self._embeddings_to_cffi(embeddings)

        result = lib.hnsw_add(self._index, arr, n, dim)

        if result != 0:
            if result == -5:
                raise DimensionMismatchError("Dimension mismatch with existing data")
            elif result == -1:
                raise HNSWError("Memory allocation failed")
            else:
                raise HNSWError(f"Failed to append embeddings (error code: {result})")

        self._dimension = dim

        # Auto-save if we have a save path
        if self._save_path:
            self.save(self._save_path)

        return self

    def close(self) -> None:
        """
        Clear all in-memory data.

        If the HNSW index was not saved, temporary disk data is deleted.
        If the HNSW index was saved, the saved file remains intact.

        Example:
            >>> db.close()  # Clears memory, keeps saved file
        """
        if not self._is_started:
            return

        # Delete temp file if not saved
        if not self._is_saved and self._temp_path and os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except:
                pass

        # Destroy the C index
        if self._index is not None:
            lib.hnsw_destroy(self._index)
            self._index = None

        self._is_started = False
        self._is_saved = False
        self._save_path = None
        self._temp_path = None
        self._dimension = None

    def clear(self) -> None:
        """
        Delete the HNSW file from disk permanently and clear all data.

        This removes both the in-memory data AND the saved file on disk.

        Example:
            >>> db.clear()  # Deletes everything, including saved file
        """
        # Delete saved file if exists
        if self._save_path and os.path.exists(self._save_path):
            try:
                os.remove(self._save_path)
            except Exception as e:
                raise FileIOError(f"Failed to delete file: {e}")

        # Delete temp file if exists
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except:
                pass

        # Destroy the C index
        if self._index is not None:
            lib.hnsw_destroy(self._index)
            self._index = None

        self._is_started = False
        self._is_saved = False
        self._save_path = None
        self._temp_path = None
        self._dimension = None

    # ========== Properties ==========

    @property
    def count(self) -> int:
        """Number of vectors in the database"""
        if not self._is_started or self._index is None:
            return 0
        return lib.hnsw_count(self._index)

    @property
    def dimension(self) -> Optional[int]:
        """Dimension of vectors in the database (None if empty)"""
        if not self._is_started or self._index is None:
            return None
        dim = lib.hnsw_dimension(self._index)
        return dim if dim > 0 else None

    @property
    def is_saved(self) -> bool:
        """Whether the database has been saved to disk"""
        return self._is_saved

    @property
    def save_path(self) -> Optional[str]:
        """Path where the database is saved (None if not saved)"""
        return self._save_path

    def get_all_vectors(self) -> List[List[float]]:
        """
        Get all vectors stored in the database.

        Returns:
            List of all embeddings
        """
        self._check_started()

        count = lib.hnsw_count(self._index)
        if count <= 0:
            return []

        total_count = ffi.new("int32_t*")
        total_dim = ffi.new("int32_t*")
        lib.hnsw_get_vectors(self._index, ffi.NULL, total_count, total_dim)

        all_vectors = ffi.new(f"float[{total_count[0] * total_dim[0]}]")
        lib.hnsw_get_vectors(self._index, all_vectors, total_count, total_dim)

        return self._cffi_to_embeddings(all_vectors, total_count[0], total_dim[0])

    # ========== Context Manager ==========

    def __enter__(self) -> 'HNSWDatabase':
        """Context manager entry"""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.close()

    def __len__(self) -> int:
        """Return number of vectors"""
        return self.count

    def __repr__(self) -> str:
        status = "started" if self._is_started else "not started"
        saved = ", saved" if self._is_saved else ""
        count = self.count if self._is_started else 0
        dim = self.dimension if self._is_started else None
        return f"HNSWDatabase({status}, count={count}, dim={dim}{saved})"
