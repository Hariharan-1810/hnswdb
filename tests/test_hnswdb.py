"""
Tests for HNSWDB package
Run with: pytest tests/test_hnswdb.py
"""

import os
import tempfile
import pytest


class TestHNSWDatabase:
    """Test suite for HNSWDatabase class"""

    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing"""
        return [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.2, 0.3, 0.4, 0.5],
            [0.9, 0.8, 0.7, 0.6],
            [0.3, 0.4, 0.5, 0.6],
        ]

    @pytest.fixture
    def db(self):
        """Create a fresh database for each test"""
        import hnswdb
        database = hnswdb.HNSWDatabase()
        database.start()
        yield database
        database.close()

    def test_start(self, db):
        """Test database initialization"""
        assert db.count == 0
        assert db.dimension is None

    def test_store(self, db, sample_embeddings):
        """Test storing embeddings"""
        db.store(sample_embeddings)
        assert db.count == 5
        assert db.dimension == 4

    def test_find(self, db, sample_embeddings):
        """Test finding nearest neighbors"""
        db.store(sample_embeddings)

        query = [0.15, 0.25, 0.35, 0.45]
        neighbors = db.find(query, k=2)

        assert len(neighbors) == 2
        assert len(neighbors[0]) == 4

    def test_find_with_distances(self, db, sample_embeddings):
        """Test finding with distances"""
        db.store(sample_embeddings)

        query = [0.1, 0.2, 0.3, 0.4]
        results = db.find_with_distances(query, k=3)

        assert len(results) == 3
        # First result should be closest (smallest distance)
        for emb, dist in results:
            assert isinstance(dist, float)
            assert dist >= 0

    def test_find_ids(self, db, sample_embeddings):
        """Test finding IDs with distances"""
        db.store(sample_embeddings)

        query = [0.1, 0.2, 0.3, 0.4]
        results = db.find_ids(query, k=3)

        assert len(results) == 3
        for idx, dist in results:
            assert isinstance(idx, int)
            assert 0 <= idx < 5

    def test_save_and_open(self, sample_embeddings):
        """Test saving and opening database"""
        import hnswdb

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.hnsw")

            # Create and save
            db1 = hnswdb.HNSWDatabase()
            db1.start()
            db1.store(sample_embeddings)
            saved_path = db1.save(filepath)
            db1.close()

            assert os.path.exists(saved_path)

            # Open and verify
            db2 = hnswdb.HNSWDatabase()
            db2.start()
            db2.open(saved_path)

            assert db2.count == 5
            assert db2.dimension == 4

            db2.close()

    def test_append(self, sample_embeddings):
        """Test appending to existing database"""
        import hnswdb

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.hnsw")

            # Create and save
            db = hnswdb.HNSWDatabase()
            db.start()
            db.store(sample_embeddings)
            db.save(filepath)

            # Append
            new_embeddings = [[0.7, 0.8, 0.9, 1.0]]
            db.append(new_embeddings)

            assert db.count == 6

            db.close()

    def test_close_without_save(self, db, sample_embeddings):
        """Test that unsaved data is cleared on close"""
        db.store(sample_embeddings)
        assert db.count == 5

        db.close()
        assert db.count == 0

    def test_clear(self, sample_embeddings):
        """Test clearing deletes saved file"""
        import hnswdb

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.hnsw")

            db = hnswdb.HNSWDatabase()
            db.start()
            db.store(sample_embeddings)
            saved_path = db.save(filepath)

            assert os.path.exists(saved_path)

            db.clear()

            assert not os.path.exists(saved_path)
            assert db.count == 0

    def test_auto_increment_filename(self, sample_embeddings):
        """Test that duplicate filenames are auto-incremented"""
        import hnswdb

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.hnsw")

            # Create first file
            db1 = hnswdb.HNSWDatabase()
            db1.start()
            db1.store(sample_embeddings)
            path1 = db1.save(filepath)
            db1.close()

            # Create second file with same name
            db2 = hnswdb.HNSWDatabase()
            db2.start()
            db2.store(sample_embeddings)
            path2 = db2.save(filepath)
            db2.close()

            assert path1 != path2
            assert os.path.exists(path1)
            assert os.path.exists(path2)

    def test_dimension_mismatch(self, db, sample_embeddings):
        """Test that dimension mismatch raises error"""
        import hnswdb

        db.store(sample_embeddings)  # dim=4

        # Try to append with wrong dimension
        wrong_dim = [[0.1, 0.2, 0.3]]  # dim=3

        with pytest.raises(hnswdb.DimensionMismatchError):
            db.append(wrong_dim)

    def test_context_manager(self, sample_embeddings):
        """Test context manager functionality"""
        import hnswdb

        with hnswdb.HNSWDatabase() as db:
            db.store(sample_embeddings)
            assert db.count == 5

        # After exiting context, should be closed
        assert db.count == 0

    def test_empty_find(self, db):
        """Test finding on empty database"""
        query = [0.1, 0.2, 0.3, 0.4]
        results = db.find(query, k=5)
        assert results == []

    def test_get_all_vectors(self, db, sample_embeddings):
        """Test retrieving all vectors"""
        db.store(sample_embeddings)

        vectors = db.get_all_vectors()
        assert len(vectors) == 5
        assert len(vectors[0]) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
