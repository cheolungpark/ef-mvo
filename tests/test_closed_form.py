"""해석적 closed-form 검증 — 최적화기가 교과서 공식과 일치하는지.

제약이 예산(과 목표수익)뿐이고 숏을 허용하면 평균-분산 최적화는
닫힌 형태 해를 가진다. 이를 기준으로 수치해를 검증한다.
"""

import numpy as np

from ef_mvo import efficient_frontier, max_return, min_variance


def _gmv_closed_form(cov):
    """전역 최소분산: w = Σ⁻¹1 / (1ᵀΣ⁻¹1)."""
    ones = np.ones(cov.shape[0])
    w = np.linalg.solve(cov, ones)
    return w / w.sum()


def _frontier_closed_form(mu, cov, r):
    """예산+목표수익 제약 최소분산의 닫힌 형태(숏 허용)."""
    ones = np.ones(len(mu))
    inv_o = np.linalg.solve(cov, ones)
    inv_m = np.linalg.solve(cov, mu)
    a = ones @ inv_o
    b = ones @ inv_m
    c = mu @ inv_m
    # 라그랑주 승수 해
    denom = a * c - b * b
    lam = (c - b * r) / denom
    gam = (a * r - b) / denom
    return lam * inv_o + gam * inv_m


def test_gmv_matches_closed_form(mu, cov):
    r = min_variance(mu, cov, bounds=(None, None))  # 숏 허용 → 예산 제약만
    np.testing.assert_allclose(r.weights, _gmv_closed_form(cov), atol=1e-4)


def test_frontier_point_matches_closed_form(mu, cov):
    for target in [0.035, 0.045, 0.055, 0.065]:
        r = min_variance(mu, cov, target_return=target, equality_return=True,
                         bounds=(None, None))
        w_cf = _frontier_closed_form(mu, cov, target)
        np.testing.assert_allclose(r.weights, w_cf, atol=1e-4)
        assert abs(r.ret - target) < 1e-8


def test_frontier_variance_is_convex_in_return(mu, cov):
    """숏 허용 프론티어의 분산은 수익에 대해 볼록(포물선)."""
    rets = np.array([0.035, 0.045, 0.055, 0.065])
    vars = []
    for t in rets:
        r = min_variance(mu, cov, target_return=t, equality_return=True,
                         bounds=(None, None))
        vars.append(r.variance)
    # 2차 차분 > 0 (볼록)
    d2 = np.diff(vars, 2)
    assert (d2 > -1e-12).all()


def test_max_return_upper_bound(mu, cov):
    r = max_return(mu, cov, bounds=(0.0, 1.0))
    assert r.ret <= float(np.max(mu)) + 1e-9
    assert r.ret == np.max(mu) or abs(r.ret - np.max(mu)) < 1e-4
