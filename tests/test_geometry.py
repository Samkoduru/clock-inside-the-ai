import numpy as np

from calendar_spike import angles, circ_score, perm_p, plane
from dial_spike import ring_model


def synthetic_ring(n=7, d=64, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(d, 2)))
    th = 2 * np.pi * np.arange(n) / n
    return np.c_[np.cos(th), np.sin(th)] @ basis.T + noise * rng.normal(size=(n, d))


def test_perfect_ring_scores_one():
    X = synthetic_ring()
    mu, b = plane(X)
    assert circ_score(angles(X, mu, b)) > 0.999


def test_ring_score_invariant_to_rotation_and_reflection():
    X = synthetic_ring(noise=0.05)
    mu, b = plane(X)
    th = angles(X, mu, b)
    assert abs(circ_score(th) - circ_score(-th + 1.234)) < 1e-9


def test_shuffled_labels_score_lower_and_p_is_small_for_true_order():
    X = synthetic_ring(noise=0.05)
    mu, b = plane(X)
    th = angles(X, mu, b)
    obs = circ_score(th)
    p = perm_p(lambda perm: circ_score(th[perm]), 7, obs)
    assert p < 0.01
    assert circ_score(th[np.array([0, 3, 6, 2, 5, 1, 4])]) < obs


def test_ring_model_reproduces_class_means_at_integer_positions():
    rng = np.random.default_rng(1)
    M = rng.normal(size=(7, 32))
    mu, Q, P = ring_model(M, [1, 2, 3])
    for i in range(7):
        assert np.allclose(P(i), M[i], atol=1e-8)
        assert np.allclose(P(i + 7), M[i], atol=1e-8)  # periodic


def test_ring_model_subspace_is_orthonormal():
    M = np.random.default_rng(2).normal(size=(7, 32))
    _, Q, _ = ring_model(M, [1, 2, 3])
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-8)
