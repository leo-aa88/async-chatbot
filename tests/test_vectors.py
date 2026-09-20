"""Unit tests for deterministic vector helpers (semantic dominance)."""

from __future__ import annotations

from aca.cognition.vectors import cosine, dominant_cluster_fraction


def test_cosine_basic():
    assert cosine([1, 0, 0], [1, 0, 0]) == 1.0
    assert cosine([1, 0, 0], [0, 1, 0]) == 0.0
    assert cosine([1, 0], [2, 0]) == 1.0  # scale-invariant
    assert cosine([0, 0], [1, 1]) == 0.0  # zero vector
    assert cosine([1, 0], [1, 0, 0]) == 0.0  # mismatched dims


def test_dominant_cluster_fraction():
    a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    # 3 in cluster A, 1 in cluster B -> largest 3 of 4.
    assert dominant_cluster_fraction([a, a, a, b], 0.78) == (3, 4)
    # All distinct neighborhoods -> largest is 1.
    assert dominant_cluster_fraction([a, b, [0.0, 0.0, 1.0]], 0.78) == (1, 3)
    # Empty / singleton.
    assert dominant_cluster_fraction([], 0.78) == (0, 0)
    assert dominant_cluster_fraction([a], 0.78) == (1, 1)


def test_threshold_controls_grouping():
    near = [0.9, 0.1, 0.0]
    a = [1.0, 0.0, 0.0]
    # cosine(a, near) ~ 0.994 -> grouped at 0.78, split at 0.999.
    assert dominant_cluster_fraction([a, near], 0.78) == (2, 2)
    assert dominant_cluster_fraction([a, near], 0.999) == (1, 2)
