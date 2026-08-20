"""MVO 엔진 단위 테스트 (수학적 성질 검증)."""

import numpy as np
import pytest

from ef_mvo import (
    cov_from_stdev_corr,
    efficient_frontier,
    indices_of,
    max_return,
    min_variance,
    portfolio_return,
    portfolio_variance,
    portfolio_volatility,
    resample_michaud,
)


# --- 기본 통계 -------------------------------------------------------------
def test_portfolio_stats_basic():
    w = [0.5, 0.5]
    mu = [0.10, 0.02]
    cov = np.array([[0.04, 0.0], [0.0, 0.01]])
    assert portfolio_return(w, mu) == pytest.approx(0.06)
    # 0.25*0.04 + 0.25*0.01 = 0.0125
    assert portfolio_variance(w, cov) == pytest.approx(0.0125)
    assert portfolio_volatility(w, cov) == pytest.approx(np.sqrt(0.0125))


def test_cov_from_stdev_corr():
    stdev = [0.2, 0.1]
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    cov = cov_from_stdev_corr(stdev, corr)
    assert cov[0, 0] == pytest.approx(0.04)
    assert cov[1, 1] == pytest.approx(0.01)
    assert cov[0, 1] == pytest.approx(0.2 * 0.1 * 0.5)


# --- 최소분산 --------------------------------------------------------------
def test_min_variance_budget_and_bounds(mu, cov):
    res = min_variance(mu, cov, bounds=(0.0, 1.0))
    assert res.success
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (res.weights >= -1e-8).all()


def test_min_variance_is_global_minimum(mu, cov):
    """무작위 실행가능 비중의 분산이 GMV 분산 이상이어야 한다."""
    gmv = min_variance(mu, cov, bounds=(0.0, 1.0))
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.random(len(mu))
        w /= w.sum()  # 예산제약 만족, 롱온리
        assert portfolio_variance(w, cov) >= gmv.variance - 1e-10


def test_target_return_is_binding(mu, cov):
    gmv = min_variance(mu, cov, bounds=(0.0, 1.0))
    target = gmv.ret + 0.01
    res = min_variance(mu, cov, target_return=target, bounds=(0.0, 1.0))
    assert res.success
    assert res.ret >= target - 1e-6
    # 목표수익을 올리면 분산은 증가(효율적 프론티어의 볼록성)
    assert res.variance >= gmv.variance - 1e-10


def test_linear_ineq_constraint(mu, cov):
    # w0 >= 0.4 를 aᵀw ≥ b 형태로 부여 (자산 수 n 에 무관)
    a = np.zeros(len(mu))
    a[0] = 1.0
    res = min_variance(mu, cov, bounds=(0.0, 1.0), linear_ineq=[(a, 0.4)])
    assert res.success
    assert res.weights[0] >= 0.4 - 1e-6


def test_exclude_pins_asset_to_zero(mu, cov):
    """exclude 한 자산은 비중 0, 나머지 합=1, 차원·순서 유지."""
    res = min_variance(mu, cov, target_return=0.05, bounds=(0.0, 1.0), exclude=[3])
    assert res.success
    assert len(res.weights) == len(mu)              # 차원 유지
    assert res.weights[3] == pytest.approx(0.0, abs=1e-9)
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_exclude_matches_manual_removal(mu, cov):
    """exclude(0 고정)가 자산 제거(n-1) 결과와 일치."""
    drop = 3
    # 방식 A: 실제 제거
    keep = [i for i in range(len(mu)) if i != drop]
    a = min_variance(mu[keep], cov[np.ix_(keep, keep)], target_return=0.05,
                     bounds=(0.0, 1.0))
    # 방식 B: exclude
    b = min_variance(mu, cov, target_return=0.05, bounds=(0.0, 1.0), exclude=[drop])
    np.testing.assert_allclose(b.weights[keep], a.weights, atol=1e-5)


def test_indices_of_names_and_ints():
    names = ["국내주식", "국내채권", "해외선진주식", "해외신흥주식", "해외채권", "현금성"]
    assert indices_of(names, ["해외신흥주식"]) == [3]
    assert indices_of(names, ["국내주식", 4]) == [0, 4]      # 이름·인덱스 혼용
    with pytest.raises(KeyError):
        indices_of(names, ["없는자산"])


def test_exclude_in_frontier_and_resample(mu, cov):
    pts = efficient_frontier(mu, cov, n_points=6, bounds=(0.0, 1.0), exclude=[3])
    assert len(pts) >= 4
    assert all(abs(p.weights[3]) < 1e-8 for p in pts)
    stdev = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stdev, stdev)
    rs = resample_michaud(mu, stdev, corr, n_sim=15, seed=1, bounds=(0.0, 1.0),
                          target_return=0.05, exclude=[3])
    assert rs.weights[3] == pytest.approx(0.0, abs=1e-8)


def test_linear_ineq_le_caps_asset(mu, cov):
    """linear_ineq_le 로 w0 ≤ 0.2 상한을 건다."""
    a = np.zeros(len(mu)); a[0] = 1.0
    res = min_variance(mu, cov, bounds=(0.0, 1.0), linear_ineq_le=[(a, 0.2)])
    assert res.success
    assert res.weights[0] <= 0.2 + 1e-6


def test_linear_ineq_le_equals_manual_sign_flip(mu, cov):
    """aᵀw ≤ c (linear_ineq_le) == (−a)ᵀw ≥ −c (linear_ineq)."""
    a = np.zeros(len(mu)); a[2] = 1.0; a[3] = -1.0   # 선진 ≤ 신흥
    le = min_variance(mu, cov, target_return=0.05, bounds=(0.0, 1.0),
                      linear_ineq_le=[(a, 0.0)])
    ge = min_variance(mu, cov, target_return=0.05, bounds=(0.0, 1.0),
                      linear_ineq=[(-a, 0.0)])
    np.testing.assert_allclose(le.weights, ge.weights, atol=1e-6)


def test_max_return_picks_highest(mu, cov):
    res = max_return(mu, cov, bounds=(0.0, 1.0))
    assert res.success
    # 최대수익 = 최고 μ 자산에 집중
    assert res.ret == pytest.approx(float(np.max(mu)), abs=1e-4)


# --- 효율적 프론티어 -------------------------------------------------------
def test_efficient_frontier_monotonic(mu, cov):
    pts = efficient_frontier(mu, cov, n_points=15, bounds=(0.0, 1.0))
    assert len(pts) >= 10
    rets = [p.ret for p in pts]
    vols = [p.volatility for p in pts]
    # 수익 오름차순 → 변동성도 비감소 (효율적 프론티어)
    assert rets == sorted(rets)
    for a, b in zip(vols, vols[1:]):
        assert b >= a - 1e-6
    # 모든 점이 예산제약 만족
    for p in pts:
        assert p.weights.sum() == pytest.approx(1.0, abs=1e-6)


# --- 리샘플링 --------------------------------------------------------------
def test_resample_michaud_runs_and_is_seeded(mu, cov):
    stdev = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stdev, stdev)
    r1 = resample_michaud(mu, stdev, corr, n_sim=20, n_samples=80, seed=42,
                          bounds=(0.0, 1.0))
    r2 = resample_michaud(mu, stdev, corr, n_sim=20, n_samples=80, seed=42,
                          bounds=(0.0, 1.0))
    assert r1.weights.sum() == pytest.approx(1.0, abs=1e-6)
    # 동일 시드 → 재현 가능
    np.testing.assert_allclose(r1.weights, r2.weights, atol=1e-12)
    # 다른 시드 → 대체로 다른 결과
    r3 = resample_michaud(mu, stdev, corr, n_sim=20, n_samples=80, seed=7,
                          bounds=(0.0, 1.0))
    assert not np.allclose(r1.weights, r3.weights, atol=1e-9)
