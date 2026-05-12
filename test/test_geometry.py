"""Unit tests for geometry utilities, especially iter_directed_crossing_pairs."""

import numpy as np

from utils.geometry import iter_directed_crossing_pairs


class TestIterDirectedCrossingPairs:
    """Test the iter_directed_crossing_pairs helper function."""

    def test_iter_directed_crossing_pairs_basic(self) -> None:
        """Test basic expansion of a single crossing pair."""
        # Create a (1, 4) array: one crossing pair (a=2, b=1, c=4, d=3)
        crossing = np.array([[2, 1, 4, 3]], dtype=np.int64)
        result = list(iter_directed_crossing_pairs(crossing))

        # Expect exactly 4 tuples in order: F/F, R/R, F/R, R/F
        expected = [
            ((2, 1), (4, 3)),  # F/F
            ((1, 2), (3, 4)),  # R/R
            ((2, 1), (3, 4)),  # F/R
            ((1, 2), (4, 3)),  # R/F
        ]
        assert result == expected, f"Expected {expected}, got {result}"

    def test_iter_directed_crossing_pairs_empty(self) -> None:
        """Test with an empty crossing array."""
        crossing = np.empty((0, 4), dtype=np.int64)
        result = list(iter_directed_crossing_pairs(crossing))
        assert result == [], f"Expected empty list, got {result}"

    def test_iter_directed_crossing_pairs_multiple(self) -> None:
        """Test with multiple crossing pairs."""
        # Two crossing pairs
        crossing = np.array(
            [
                [2, 1, 4, 3],
                [5, 0, 7, 6],
            ],
            dtype=np.int64,
        )
        result = list(iter_directed_crossing_pairs(crossing))

        # Expect 8 tuples (4 per row)
        assert len(result) == 8, f"Expected 8 tuples, got {len(result)}"

        # First 4 are from row 0
        assert result[0] == ((2, 1), (4, 3)), f"Row 0, combo 0: got {result[0]}"
        assert result[1] == ((1, 2), (3, 4)), f"Row 0, combo 1: got {result[1]}"
        assert result[2] == ((2, 1), (3, 4)), f"Row 0, combo 2: got {result[2]}"
        assert result[3] == ((1, 2), (4, 3)), f"Row 0, combo 3: got {result[3]}"

        # Next 4 are from row 1
        assert result[4] == ((5, 0), (7, 6)), f"Row 1, combo 0: got {result[4]}"
        assert result[5] == ((0, 5), (6, 7)), f"Row 1, combo 1: got {result[5]}"
        assert result[6] == ((5, 0), (6, 7)), f"Row 1, combo 2: got {result[6]}"
        assert result[7] == ((0, 5), (7, 6)), f"Row 1, combo 3: got {result[7]}"

    def test_iter_directed_crossing_pairs_order(self) -> None:
        """Test that yield order is deterministic and correct."""
        crossing = np.array([[10, 5, 8, 3]], dtype=np.int64)
        result = list(iter_directed_crossing_pairs(crossing))

        # Verify order: F/F, R/R, F/R, R/F
        assert result[0] == ((10, 5), (8, 3)), "First should be F/F"
        assert result[1] == ((5, 10), (3, 8)), "Second should be R/R"
        assert result[2] == ((10, 5), (3, 8)), "Third should be F/R"
        assert result[3] == ((5, 10), (8, 3)), "Fourth should be R/F"

    def test_iter_directed_crossing_pairs_is_generator(self) -> None:
        """Test that the function returns a generator, not a list."""
        crossing = np.array([[2, 1, 4, 3]], dtype=np.int64)
        result = iter_directed_crossing_pairs(crossing)

        # Should be a generator
        assert hasattr(result, "__iter__"), "Result should be iterable"
        assert hasattr(result, "__next__"), "Result should be a generator"

        # Can iterate multiple times by creating new generator
        result2 = iter_directed_crossing_pairs(crossing)
        list1 = list(result2)
        result3 = iter_directed_crossing_pairs(crossing)
        list2 = list(result3)
        assert list1 == list2, "Two fresh generators should yield same values"
