"""
Comprehensive Test Suite for HNSWDB

This test file covers all functionality of the HNSWDB package:
1. Database lifecycle (start, close, context manager)
2. Storing embeddings (single, batch, various dimensions)
3. Finding neighbors (find, find_with_distances, find_ids)
4. File operations (save, open, append)
5. Edge cases and error handling
6. Correctness verification (actually finds closest)

Run with: pytest test_comprehensive.py -v
Or:       python test_comprehensive.py

Requirements:
- pytest (pip install pytest)
- hnswdb must be built (python hnswdb/build_cffi.py or pip install .)
"""

import os
import sys
import math
import tempfile
import shutil
from typing import List, Tuple

# Try to import pytest, but allow running standalone
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Mock pytest decorators for standalone execution
    class MockPytest:
        @staticmethod
        def fixture(func=None, **kwargs):
            if func:
                return func
            return lambda f: f

        @staticmethod
        def mark(*args, **kwargs):
            class Mark:
                @staticmethod
                def parametrize(*a, **kw):
                    return lambda f: f
            return Mark()

        @staticmethod
        def raises(exc):
            class RaisesContext:
                def __enter__(self): return self
                def __exit__(self, *args): return True
            return RaisesContext()

    pytest = MockPytest()

# Import hnswdb
try:
    import hnswdb
    from hnswdb import HNSWDatabase, HNSWError, DimensionMismatchError, NotInitializedError, FileIOError
    HNSWDB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import hnswdb: {e}")
    print("Make sure to build the package first: pip install -e .")
    HNSWDB_AVAILABLE = False


# ============================================================
# Test Utilities
# ============================================================

def cosine_distance(a: List[float], b: List[float]) -> float:
    """Calculate cosine distance between two vectors (for verification)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    return 1.0 - similarity


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Calculate Euclidean distance."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def generate_random_embeddings(n: int, dim: int, seed: int = 42) -> List[List[float]]:
    """Generate random embeddings for testing."""
    import random
    random.seed(seed)
    return [[random.random() for _ in range(dim)] for _ in range(n)]


def normalize_vector(v: List[float]) -> List[float]:
    """Normalize a vector to unit length."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-10:
        return v
    return [x / norm for x in v]


def brute_force_knn(query: List[float], embeddings: List[List[float]], k: int) -> List[Tuple[int, float]]:
    """Brute force k-nearest neighbors for correctness verification."""
    distances = [(i, cosine_distance(query, emb)) for i, emb in enumerate(embeddings)]
    distances.sort(key=lambda x: x[1])
    return distances[:k]


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def sample_embeddings_small():
    """Small set of embeddings for basic tests."""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.2, 0.3, 0.4, 0.5],
        [0.9, 0.8, 0.7, 0.6],
        [0.3, 0.4, 0.5, 0.6],
    ]


@pytest.fixture
def sample_embeddings_medium():
    """Medium set of embeddings."""
    return generate_random_embeddings(100, 64)


@pytest.fixture
def sample_embeddings_384d():
    """Embeddings similar to common sentence-embedding dimension."""
    return generate_random_embeddings(50, 384)


@pytest.fixture
def db():
    """Create a fresh database for each test."""
    if not HNSWDB_AVAILABLE:
        pytest.skip("hnswdb not available")
    database = HNSWDatabase()
    database.start()
    yield database
    try:
        database.close()
    except:
        pass


# ============================================================
# Test Classes
# ============================================================

class TestDatabaseLifecycle:
    """Tests for database initialization and cleanup."""

    def test_create_database(self):
        """Test creating a new database instance."""
        if not HNSWDB_AVAILABLE:
            return
        db = HNSWDatabase()
        assert db is not None
        assert db.count == 0

    def test_start_database(self):
        """Test starting the database."""
        if not HNSWDB_AVAILABLE:
            return
        db = HNSWDatabase()
        result = db.start()
        assert result is db  # Method chaining
        assert db.count == 0
        assert db.dimension is None
        db.close()

    def test_double_start(self):
        """Test that double start is safe."""
        if not HNSWDB_AVAILABLE:
            return
        db = HNSWDatabase()
        db.start()
        db.start()  # Should not raise
        assert db.count == 0
        db.close()

    def test_close_database(self):
        """Test closing the database."""
        if not HNSWDB_AVAILABLE:
            return
        db = HNSWDatabase()
        db.start()
        db.close()
        # Should be safe to close again
        db.close()

    def test_context_manager(self, sample_embeddings_small):
        """Test using database as context manager."""
        if not HNSWDB_AVAILABLE:
            return
        with HNSWDatabase() as db:
            db.store(sample_embeddings_small)
            assert db.count == 5
        # After context, db should be closed

    def test_operations_before_start(self):
        """Test that operations fail before start."""
        if not HNSWDB_AVAILABLE:
            return
        db = HNSWDatabase()
        try:
            db.store([[0.1, 0.2, 0.3]])
            assert False, "Should have raised NotInitializedError"
        except NotInitializedError:
            pass


class TestStoreEmbeddings:
    """Tests for storing embeddings."""

    def test_store_basic(self, db, sample_embeddings_small):
        """Test basic embedding storage."""
        db.store(sample_embeddings_small)
        assert db.count == 5
        assert db.dimension == 4

    def test_store_single_embedding(self, db):
        """Test storing a single embedding."""
        db.store([[1.0, 2.0, 3.0]])
        assert db.count == 1
        assert db.dimension == 3

    def test_store_large_dimension(self, db, sample_embeddings_384d):
        """Test storing high-dimensional embeddings."""
        db.store(sample_embeddings_384d)
        assert db.count == 50
        assert db.dimension == 384

    def test_store_method_chaining(self, db):
        """Test that store returns self for chaining."""
        result = db.store([[0.1, 0.2, 0.3]])
        assert result is db

    def test_store_empty_raises(self, db):
        """Test that storing empty list raises error."""
        try:
            db.store([])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "empty" in str(e).lower()

    def test_store_dimension_mismatch(self, db):
        """Test that mismatched dimensions raise error."""
        db.store([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        try:
            db.store([[1.0, 2.0]])  # Different dimension
            assert False, "Should have raised DimensionMismatchError"
        except DimensionMismatchError:
            pass

    def test_store_inconsistent_dimensions(self, db):
        """Test that inconsistent dimensions in same batch raise error."""
        try:
            db.store([[1.0, 2.0, 3.0], [1.0, 2.0]])  # Different dims
            assert False, "Should have raised DimensionMismatchError"
        except DimensionMismatchError:
            pass

    def test_store_auto_detect_dimension(self, db):
        """Test that dimension is auto-detected."""
        assert db.dimension is None
        db.store([[1.0, 2.0, 3.0, 4.0, 5.0]])
        assert db.dimension == 5

    def test_store_multiple_batches(self, db):
        """Test storing multiple batches of embeddings."""
        db.store([[1.0, 2.0, 3.0]])
        assert db.count == 1
        db.store([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        assert db.count == 3


class TestFindNeighbors:
    """Tests for finding nearest neighbors."""

    def test_find_basic(self, db, sample_embeddings_small):
        """Test basic neighbor finding."""
        db.store(sample_embeddings_small)
        query = [0.15, 0.25, 0.35, 0.45]
        neighbors = db.find(query, k=2)
        assert len(neighbors) == 2
        assert len(neighbors[0]) == 4

    def test_find_returns_vectors(self, db, sample_embeddings_small):
        """Test that find returns actual vectors."""
        db.store(sample_embeddings_small)
        neighbors = db.find(sample_embeddings_small[0], k=1)
        assert len(neighbors) == 1
        # The closest neighbor to embedding[0] should be itself
        for i, val in enumerate(neighbors[0]):
            assert abs(val - sample_embeddings_small[0][i]) < 0.001

    def test_find_k_larger_than_count(self, db):
        """Test finding when k > number of embeddings."""
        db.store([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        neighbors = db.find([1.0, 2.0, 3.0], k=10)
        assert len(neighbors) == 2  # Only 2 available

    def test_find_k_equals_one(self, db, sample_embeddings_small):
        """Test finding single nearest neighbor."""
        db.store(sample_embeddings_small)
        neighbors = db.find(sample_embeddings_small[0], k=1)
        assert len(neighbors) == 1

    def test_find_with_distances(self, db, sample_embeddings_small):
        """Test finding with distance values."""
        db.store(sample_embeddings_small)
        results = db.find_with_distances(sample_embeddings_small[0], k=3)
        assert len(results) == 3
        for emb, dist in results:
            assert isinstance(emb, list)
            assert isinstance(dist, float)
            assert dist >= 0

    def test_find_distances_sorted(self, db, sample_embeddings_medium):
        """Test that results are sorted by distance (ascending)."""
        db.store(sample_embeddings_medium)
        query = sample_embeddings_medium[0]
        results = db.find_with_distances(query, k=10)
        distances = [d for _, d in results]
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1], "Results not sorted"

    def test_find_ids(self, db, sample_embeddings_small):
        """Test finding by ID."""
        db.store(sample_embeddings_small)
        results = db.find_ids(sample_embeddings_small[0], k=3)
        assert len(results) == 3
        for idx, dist in results:
            assert isinstance(idx, int)
            assert 0 <= idx < 5
            assert isinstance(dist, float)

    def test_find_empty_index(self, db):
        """Test finding on empty index."""
        neighbors = db.find([1.0, 2.0, 3.0], k=5)
        assert neighbors == []

    def test_find_dimension_mismatch(self, db, sample_embeddings_small):
        """Test finding with wrong dimension query."""
        db.store(sample_embeddings_small)  # dim=4
        try:
            db.find([1.0, 2.0], k=1)  # dim=2
            assert False, "Should have raised DimensionMismatchError"
        except DimensionMismatchError:
            pass


class TestCorrectnessVerification:
    """Tests that verify the algorithm finds the correct neighbors."""

    def test_finds_exact_match(self, db, sample_embeddings_small):
        """Test that exact match is found first."""
        db.store(sample_embeddings_small)
        for i, emb in enumerate(sample_embeddings_small):
            results = db.find_ids(emb, k=1)
            assert len(results) >= 1
            assert results[0][0] == i, f"Expected ID {i}, got {results[0][0]}"
            assert results[0][1] < 0.001, "Distance to self should be ~0"

    def test_finds_closest_neighbors(self, db):
        """Test that actual closest neighbors are found."""
        # Create embeddings with known closest pairs
        embeddings = [
            [1.0, 0.0, 0.0],  # ID 0
            [0.9, 0.1, 0.0],  # ID 1 - close to 0
            [0.0, 1.0, 0.0],  # ID 2 - orthogonal to 0
            [0.0, 0.0, 1.0],  # ID 3 - orthogonal to 0
            [-1.0, 0.0, 0.0], # ID 4 - opposite to 0
        ]
        db.store(embeddings)

        # Query for neighbors of ID 0
        results = db.find_ids([1.0, 0.0, 0.0], k=3)
        ids = [r[0] for r in results]

        # ID 0 (exact match) should be first
        assert ids[0] == 0
        # ID 1 (most similar) should be second
        assert ids[1] == 1

    def test_correctness_against_brute_force(self, db):
        """Verify HNSW results match brute force for small dataset."""
        embeddings = generate_random_embeddings(50, 16, seed=123)
        db.store(embeddings)

        # Test multiple queries
        for seed in range(10):
            query = generate_random_embeddings(1, 16, seed=1000 + seed)[0]

            # HNSW search
            hnsw_results = db.find_ids(query, k=5)
            hnsw_ids = set(r[0] for r in hnsw_results)

            # Brute force
            bf_results = brute_force_knn(query, embeddings, k=5)
            bf_ids = set(r[0] for r in bf_results)

            # Allow some approximation error but expect high overlap
            overlap = len(hnsw_ids.intersection(bf_ids))
            assert overlap >= 4, f"Low overlap: HNSW={hnsw_ids}, BF={bf_ids}"

    def test_distance_values_correct(self, db):
        """Test that returned distances are correct."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        db.store(embeddings)

        query = [1.0, 0.0, 0.0]
        results = db.find_with_distances(query, k=2)

        # First result should be exact match (distance ~0)
        assert results[0][1] < 0.001

        # Second result should be orthogonal (distance ~1.0)
        assert 0.99 < results[1][1] < 1.01


class TestFilePersistence:
    """Tests for save/load functionality."""

    def test_save_basic(self, db, sample_embeddings_small, temp_dir):
        """Test basic save operation."""
        db.store(sample_embeddings_small)
        filepath = os.path.join(temp_dir, "test.hnsw")
        saved_path = db.save(filepath)
        assert os.path.exists(saved_path)
        assert saved_path.endswith(".hnsw")

    def test_save_auto_extension(self, db, sample_embeddings_small, temp_dir):
        """Test that .hnsw extension is added automatically."""
        db.store(sample_embeddings_small)
        filepath = os.path.join(temp_dir, "test")  # No extension
        saved_path = db.save(filepath)
        assert saved_path.endswith(".hnsw")
        assert os.path.exists(saved_path)

    def test_save_unique_filename(self, db, sample_embeddings_small, temp_dir):
        """Test that existing files get numbered."""
        db.store(sample_embeddings_small)
        filepath = os.path.join(temp_dir, "test.hnsw")

        path1 = db.save(filepath)
        path2 = db.save(filepath)  # Should create test1.hnsw

        assert path1 != path2
        assert os.path.exists(path1)
        assert os.path.exists(path2)

    def test_save_empty_raises(self, db, temp_dir):
        """Test that saving empty index raises error."""
        filepath = os.path.join(temp_dir, "test.hnsw")
        try:
            db.save(filepath)
            assert False, "Should have raised HNSWError"
        except HNSWError:
            pass

    def test_open_basic(self, sample_embeddings_small, temp_dir):
        """Test basic open operation."""
        if not HNSWDB_AVAILABLE:
            return

        # Create and save
        db1 = HNSWDatabase()
        db1.start()
        db1.store(sample_embeddings_small)
        filepath = db1.save(os.path.join(temp_dir, "test.hnsw"))
        db1.close()

        # Open and verify
        db2 = HNSWDatabase()
        db2.start()
        db2.open(filepath)

        assert db2.count == 5
        assert db2.dimension == 4
        db2.close()

    def test_open_nonexistent_raises(self, db, temp_dir):
        """Test that opening nonexistent file raises error."""
        filepath = os.path.join(temp_dir, "nonexistent.hnsw")
        try:
            db.open(filepath)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_open_preserves_data(self, sample_embeddings_small, temp_dir):
        """Test that loaded data matches original."""
        if not HNSWDB_AVAILABLE:
            return

        # Create and save
        db1 = HNSWDatabase()
        db1.start()
        db1.store(sample_embeddings_small)
        filepath = db1.save(os.path.join(temp_dir, "test.hnsw"))

        # Search before close
        original_results = db1.find_ids(sample_embeddings_small[0], k=3)
        db1.close()

        # Load and search again
        db2 = HNSWDatabase()
        db2.start()
        db2.open(filepath)
        loaded_results = db2.find_ids(sample_embeddings_small[0], k=3)
        db2.close()

        # Results should match
        assert len(original_results) == len(loaded_results)
        for (id1, d1), (id2, d2) in zip(original_results, loaded_results):
            assert id1 == id2
            assert abs(d1 - d2) < 0.001

    def test_append_and_autosave(self, sample_embeddings_small, temp_dir):
        """Test appending to loaded database."""
        if not HNSWDB_AVAILABLE:
            return

        # Create and save
        db1 = HNSWDatabase()
        db1.start()
        db1.store(sample_embeddings_small[:3])
        filepath = db1.save(os.path.join(temp_dir, "test.hnsw"))
        db1.close()

        # Open, append, close
        db2 = HNSWDatabase()
        db2.start()
        db2.open(filepath)
        assert db2.count == 3
        db2.append(sample_embeddings_small[3:])
        assert db2.count == 5
        db2.close()

        # Verify persistence
        db3 = HNSWDatabase()
        db3.start()
        db3.open(filepath)
        assert db3.count == 5
        db3.close()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_embedding(self, db):
        """Test with single embedding."""
        db.store([[1.0, 2.0, 3.0]])
        results = db.find([1.0, 2.0, 3.0], k=1)
        assert len(results) == 1

    def test_identical_embeddings(self, db):
        """Test with multiple identical embeddings."""
        emb = [1.0, 2.0, 3.0]
        db.store([emb, emb, emb, emb, emb])
        results = db.find(emb, k=5)
        assert len(results) == 5
        # All distances should be ~0
        results_with_dist = db.find_with_distances(emb, k=5)
        for _, dist in results_with_dist:
            assert dist < 0.001

    def test_zero_vector(self, db):
        """Test with zero vector."""
        db.store([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        results = db.find([0.0, 0.0, 0.0], k=2)
        assert len(results) == 2

    def test_negative_values(self, db):
        """Test with negative values."""
        db.store([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])
        results = db.find([-1.0, -2.0, -3.0], k=1)
        assert len(results) == 1

    def test_very_large_values(self, db):
        """Test with very large values."""
        emb1 = [1e6, 2e6, 3e6]
        emb2 = [1e-6, 2e-6, 3e-6]
        db.store([emb1, emb2])
        results = db.find(emb1, k=2)
        assert len(results) == 2

    def test_high_dimension(self, db):
        """Test with high dimensional vectors."""
        dim = 1536  # Common large embedding dimension
        embeddings = generate_random_embeddings(10, dim)
        db.store(embeddings)

        query = embeddings[0]
        results = db.find(query, k=3)
        assert len(results) == 3
        assert len(results[0]) == dim

    def test_many_embeddings(self, db):
        """Test with many embeddings."""
        n = 1000
        embeddings = generate_random_embeddings(n, 32)
        db.store(embeddings)

        assert db.count == n
        results = db.find(embeddings[0], k=10)
        assert len(results) == 10


class TestProperties:
    """Tests for database properties."""

    def test_count_property(self, db, sample_embeddings_small):
        """Test count property."""
        assert db.count == 0
        db.store(sample_embeddings_small)
        assert db.count == 5

    def test_dimension_property(self, db, sample_embeddings_small):
        """Test dimension property."""
        assert db.dimension is None
        db.store(sample_embeddings_small)
        assert db.dimension == 4

    def test_is_saved_property(self, db, sample_embeddings_small, temp_dir):
        """Test is_saved property."""
        db.store(sample_embeddings_small)
        assert db.is_saved == False
        db.save(os.path.join(temp_dir, "test.hnsw"))
        assert db.is_saved == True

    def test_save_path_property(self, db, sample_embeddings_small, temp_dir):
        """Test save_path property."""
        assert db.save_path is None
        db.store(sample_embeddings_small)
        filepath = os.path.join(temp_dir, "test.hnsw")
        actual = db.save(filepath)
        assert db.save_path == actual


class TestClear:
    """Tests for clear functionality."""

    def test_clear_basic(self, db, sample_embeddings_small):
        """Test clearing the database."""
        db.store(sample_embeddings_small)
        assert db.count == 5
        db.clear()
        assert db.count == 0

    def test_clear_resets_dimension(self, db, sample_embeddings_small):
        """Test that clear resets dimension."""
        db.store(sample_embeddings_small)
        assert db.dimension == 4
        db.clear()
        assert db.dimension is None

    def test_store_after_clear(self, db, sample_embeddings_small):
        """Test storing after clear."""
        db.store(sample_embeddings_small)
        db.clear()
        db.start()  # clear() fully resets the database, so restart it
        db.store([[1.0, 2.0, 3.0]])  # Different dimension
        assert db.count == 1
        assert db.dimension == 3


# ============================================================
# Standalone Test Runner
# ============================================================

def run_tests_standalone():
    """Run all tests without pytest."""
    print("=" * 60)
    print("HNSWDB Comprehensive Test Suite (Standalone)")
    print("=" * 60)

    if not HNSWDB_AVAILABLE:
        print("\nERROR: hnswdb not available. Build with: pip install -e .")
        return False

    # Create test fixtures manually
    temp_dir = tempfile.mkdtemp()
    sample_small = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.2, 0.3, 0.4, 0.5],
        [0.9, 0.8, 0.7, 0.6],
        [0.3, 0.4, 0.5, 0.6],
    ]
    sample_medium = generate_random_embeddings(100, 64)
    sample_384d = generate_random_embeddings(50, 384)

    tests_passed = 0
    tests_failed = 0
    failures = []

    # Collect all test methods
    test_classes = [
        TestDatabaseLifecycle,
        TestStoreEmbeddings,
        TestFindNeighbors,
        TestCorrectnessVerification,
        TestFilePersistence,
        TestEdgeCases,
        TestProperties,
        TestClear,
    ]

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if not method_name.startswith('test_'):
                continue

            method = getattr(instance, method_name)
            test_name = f"  {method_name}"

            try:
                # Create fresh db for each test
                db = HNSWDatabase()
                db.start()

                # Inject fixtures based on parameter names
                import inspect
                params = inspect.signature(method).parameters
                kwargs = {}

                if 'db' in params:
                    kwargs['db'] = db
                if 'sample_embeddings_small' in params:
                    kwargs['sample_embeddings_small'] = sample_small
                if 'sample_embeddings_medium' in params:
                    kwargs['sample_embeddings_medium'] = sample_medium
                if 'sample_embeddings_384d' in params:
                    kwargs['sample_embeddings_384d'] = sample_384d
                if 'temp_dir' in params:
                    kwargs['temp_dir'] = temp_dir

                method(**kwargs)

                print(f"{test_name}: PASSED")
                tests_passed += 1

            except Exception as e:
                print(f"{test_name}: FAILED - {e}")
                tests_failed += 1
                failures.append((test_name, str(e)))

            finally:
                try:
                    db.close()
                except:
                    pass

    # Cleanup temp directory
    try:
        shutil.rmtree(temp_dir)
    except:
        pass

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")

    if failures:
        print("\nFailed tests:")
        for name, error in failures:
            print(f"  {name}: {error}")

    print("=" * 60)
    return tests_failed == 0


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    if HAS_PYTEST and len(sys.argv) == 1:
        # Run with pytest if available
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        # Run standalone
        success = run_tests_standalone()
        sys.exit(0 if success else 1)
