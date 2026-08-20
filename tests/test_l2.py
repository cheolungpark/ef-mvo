"""L2(릿지) 정규화 테스트.

목적함수 ``min wᵀΣw + γ·wᵀw`` — L2 페널티로 비중을 분산시켜
추정오차 민감도를 낮춘다. 예산제약 하에서 γ·wᵀw 는 γ·‖w−1/n‖²
(등가중 편차 페널티)와 동치다.
"""

import numpy as np
import pytest

from ef_mvo import (
    efficient_frontier_l2,
    fit_l2_gamma,
    min_variance,
    min_variance_l2,
)


def test_l2_reduces_concentration(mu, cov):
    """동일 목표수익에서 γ↑ → ‖w‖²(집중도) 감소, 분산은 비감소."""
    target = 0.05
    base = min_variance(mu, cov, target_return=target, bounds=(0.0, 1.0))
    prev_conc = base.weights @ base.weights
    prev_var = base.variance
    for g in [1e-3, 5e-3, 2e-2, 1e-1]:
        r = min_variance_l2(mu, cov, g, target_return=target, bounds=(0.0, 1.0))
        conc = r.weights @ r.weights
        assert conc <= prev_conc + 1e-9          # 더 분산됨
        assert r.variance >= prev_var - 1e-9     # 순분산은 악화(또는 동일)
        assert r.ret == pytest.approx(target, abs=1e-6)
        prev_conc, prev_var = conc, r.variance


def test_l2_large_gamma_approaches_equal_weight(mu, cov):
    """γ 가 매우 크면 (예산·롱온리만 있을 때) 등가중에 수렴."""
    n = len(mu)
    r = min_variance_l2(mu, cov, 1e6, bounds=(0.0, 1.0))
    np.testing.assert_allclose(r.weights, np.repeat(1 / n, n), atol=1e-3)


def test_l2_gamma_zero_equals_min_variance(mu, cov):
    """γ=0 이면 순수 최소분산과 동일."""
    a = min_variance(mu, cov, target_return=0.05, bounds=(0.0, 1.0))
    b = min_variance_l2(mu, cov, 0.0, target_return=0.05, bounds=(0.0, 1.0))
    np.testing.assert_allclose(a.weights, b.weights, atol=1e-8)


def test_l2_closed_form_unconstrained(mu, cov):
    """제약이 예산뿐일 때 L2 해의 닫힌 형태와 일치.

    min wᵀ(Σ+γI)w  s.t. 1ᵀw=1  →  w = (Σ+γI)⁻¹1 / (1ᵀ(Σ+γI)⁻¹1)
    """
    n = len(mu)
    gamma = 0.01
    M = cov + gamma * np.eye(n)
    ones = np.ones(n)
    w_cf = np.linalg.solve(M, ones)
    w_cf /= w_cf.sum()
    r = min_variance_l2(mu, cov, gamma, bounds=(None, None))  # 숏 허용(제약=예산)
    np.testing.assert_allclose(r.weights, w_cf, atol=1e-6)


def test_l2_fit_gamma_roundtrip(mu, cov):
    """min_variance_l2(γ*) 의 해를 목표로 주면 γ* 를 되찾는다."""
    target = 0.05
    gamma_true = 0.02
    w_target = min_variance_l2(mu, cov, gamma_true, target_return=target,
                               equality_return=True, bounds=(0.0, 1.0)).weights
    gamma_fit, resid = fit_l2_gamma(mu, cov, target, w_target, bounds=(0.0, 1.0))
    assert resid < 1e-3
    assert gamma_fit == pytest.approx(gamma_true, rel=0.15)


def test_l2_frontier_is_more_diversified_than_ef(mu, cov):
    ef = efficient_frontier_l2(mu, cov, 0.0, n_points=12, bounds=(0.0, 1.0))
    l2 = efficient_frontier_l2(mu, cov, 1e-2, n_points=12, bounds=(0.0, 1.0))
    assert len(ef) >= 8 and len(l2) >= 8
    conc_ef = np.mean([p.weights @ p.weights for p in ef])
    conc_l2 = np.mean([p.weights @ p.weights for p in l2])
    assert conc_l2 < conc_ef
