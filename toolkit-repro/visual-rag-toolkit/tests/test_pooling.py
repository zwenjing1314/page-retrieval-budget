"""Tests for pooling functions."""

import numpy as np


class TestTileLevelPooling:
    """Test tile-level mean pooling."""

    def test_basic_pooling(self):
        """Pooling reduces [num_tokens, dim] → [num_tiles, dim]."""
        from visual_rag.embedding.pooling import tile_level_mean_pooling

        # 13 tiles × 64 patches = 832 visual tokens
        num_tiles = 13
        patches_per_tile = 64
        num_tokens = num_tiles * patches_per_tile
        dim = 128

        embedding = np.random.randn(num_tokens, dim).astype(np.float32)
        pooled = tile_level_mean_pooling(embedding, num_tiles, patches_per_tile)

        assert pooled.shape == (num_tiles, dim)
        assert pooled.dtype == np.float32

    def test_pooling_preserves_info(self):
        """Pooled vectors should be mean of patches."""
        from visual_rag.embedding.pooling import tile_level_mean_pooling

        num_tiles = 5
        patches_per_tile = 64
        dim = 128

        embedding = np.random.randn(num_tiles * patches_per_tile, dim).astype(np.float32)
        pooled = tile_level_mean_pooling(embedding, num_tiles, patches_per_tile)

        # Check first tile
        expected_tile0 = embedding[:patches_per_tile].mean(axis=0)
        np.testing.assert_array_almost_equal(pooled[0], expected_tile0, decimal=5)

    def test_pooling_with_partial_last_tile(self):
        """Handle case where last tile has fewer patches."""
        from visual_rag.embedding.pooling import tile_level_mean_pooling

        # 800 tokens, 64 per tile = 12.5 tiles → 13 tiles with partial last
        num_tokens = 800
        num_tiles = 13
        dim = 128

        embedding = np.random.randn(num_tokens, dim).astype(np.float32)
        pooled = tile_level_mean_pooling(embedding, num_tiles, patches_per_tile=64)

        # Should handle gracefully - at least some tiles
        assert pooled.shape[1] == dim
        assert pooled.shape[0] >= 1


class TestGlobalPooling:
    """Test global mean pooling."""

    def test_global_mean(self):
        """Global pooling reduces to single vector."""
        from visual_rag.embedding.pooling import global_mean_pooling

        embedding = np.random.randn(832, 128).astype(np.float32)
        pooled = global_mean_pooling(embedding)

        assert pooled.shape == (128,)
        np.testing.assert_array_almost_equal(pooled, embedding.mean(axis=0))


class TestColPaliExperimentalPooling:
    """Test ColPali experimental pooling (convolution-style with window 3)."""

    def test_output_shape_n_plus_2(self):
        """For N rows, should produce N + 2 vectors."""
        from visual_rag.embedding.pooling import colpali_experimental_pooling_from_rows

        for n in [4, 10, 32, 64]:
            rows = np.random.randn(n, 128).astype(np.float32)
            pooled = colpali_experimental_pooling_from_rows(rows)
            assert pooled.shape == (n + 2, 128), f"Expected ({n + 2}, 128), got {pooled.shape}"

    def test_32_rows_produces_34_vectors(self):
        """Specifically test ColPali case: 32 rows → 34 vectors."""
        from visual_rag.embedding.pooling import colpali_experimental_pooling_from_rows

        rows = np.random.randn(32, 128).astype(np.float32)
        pooled = colpali_experimental_pooling_from_rows(rows)
        assert pooled.shape == (34, 128), f"Expected (34, 128), got {pooled.shape}"

    def test_first_vector_is_first_row(self):
        """Position 0 should be first row alone."""
        from visual_rag.embedding.pooling import colpali_experimental_pooling_from_rows

        rows = np.random.randn(10, 128).astype(np.float32)
        pooled = colpali_experimental_pooling_from_rows(rows)
        np.testing.assert_array_almost_equal(pooled[0], rows[0])

    def test_second_vector_is_mean_first_two(self):
        """Position 1 should be mean of first 2 rows."""
        from visual_rag.embedding.pooling import colpali_experimental_pooling_from_rows

        rows = np.random.randn(10, 128).astype(np.float32)
        pooled = colpali_experimental_pooling_from_rows(rows)
        expected = rows[:2].mean(axis=0)
        np.testing.assert_array_almost_equal(pooled[1], expected)

    def test_third_vector_is_mean_first_three(self):
        """Position 2 should be mean of first 3 rows."""
        from visual_rag.embedding.pooling import colpali_experimental_pooling_from_rows

        rows = np.random.randn(10, 128).astype(np.float32)
        pooled = colpali_experimental_pooling_from_rows(rows)
        expected = rows[:3].mean(axis=0)
        np.testing.assert_array_almost_equal(pooled[2], expected)

    def test_sliding_window_middle(self):
        """Middle positions should be sliding window of 3."""
        from visual_rag.embedding.pooling import colpali_experimental_pooling_from_rows

        rows = np.random.randn(10, 128).astype(np.float32)
        pooled = colpali_experimental_pooling_from_rows(rows)
        expected_pos5 = rows[3:6].mean(axis=0)
        np.testing.assert_array_almost_equal(pooled[5], expected_pos5)

    def test_last_vector_is_last_row(self):
        """Last position should be last row alone."""
        from visual_rag.embedding.pooling import colpali_experimental_pooling_from_rows

        rows = np.random.randn(10, 128).astype(np.float32)
        pooled = colpali_experimental_pooling_from_rows(rows)
        np.testing.assert_array_almost_equal(pooled[-1], rows[-1])

    def test_second_to_last_is_mean_last_two(self):
        """Position N should be mean of last 2 rows."""
        from visual_rag.embedding.pooling import colpali_experimental_pooling_from_rows

        rows = np.random.randn(10, 128).astype(np.float32)
        pooled = colpali_experimental_pooling_from_rows(rows)
        expected = rows[-2:].mean(axis=0)
        np.testing.assert_array_almost_equal(pooled[-2], expected)

    def test_edge_cases(self):
        """Test edge cases: n=1, n=2, n=3."""
        from visual_rag.embedding.pooling import colpali_experimental_pooling_from_rows

        rows1 = np.random.randn(1, 128).astype(np.float32)
        pooled1 = colpali_experimental_pooling_from_rows(rows1)
        assert pooled1.shape == (1, 128)

        rows2 = np.random.randn(2, 128).astype(np.float32)
        pooled2 = colpali_experimental_pooling_from_rows(rows2)
        assert pooled2.shape == (3, 128)

        rows3 = np.random.randn(3, 128).astype(np.float32)
        pooled3 = colpali_experimental_pooling_from_rows(rows3)
        assert pooled3.shape == (5, 128)


class TestMaxSimScore:
    """Test MaxSim scoring."""

    def test_maxsim_identical(self):
        """Identical embeddings should have high score."""
        from visual_rag.embedding.pooling import compute_maxsim_score

        embedding = np.random.randn(10, 128).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        score = compute_maxsim_score(embedding, embedding)

        # Should be close to num_tokens (each token matches itself perfectly)
        assert score >= 9.0  # Allow some floating point tolerance

    def test_maxsim_orthogonal(self):
        """Orthogonal embeddings should have low score."""
        from visual_rag.embedding.pooling import compute_maxsim_score

        # Create orthogonal vectors
        query = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        doc = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

        score = compute_maxsim_score(query, doc)

        assert score < 0.1  # Near zero for orthogonal

    def test_maxsim_shape_independence(self):
        """Score should work with different query/doc lengths."""
        from visual_rag.embedding.pooling import compute_maxsim_score

        query = np.random.randn(5, 128).astype(np.float32)
        doc = np.random.randn(100, 128).astype(np.float32)

        score = compute_maxsim_score(query, doc)

        assert isinstance(score, float)
        assert not np.isnan(score)
