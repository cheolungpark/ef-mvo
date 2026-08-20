"""데모: 최소분산 · 효율적 프론티어 · L2 정규화 · 리샘플링 실행.

    python demo.py
"""

import numpy as np

from ef_mvo import (
    efficient_frontier,
    fit_l2_gamma,
    indices_of,
    min_variance,
    min_variance_l2,
    resample_michaud,
)
from ef_mvo.sample_data import sample


def main():
    mu, cov, names = sample()

    print(f"자산: {names}")

    print("\n[1] 전역 최소분산 포트폴리오 (롱온리)")
    res = min_variance(mu, cov, bounds=(0.0, 1.0))
    for n, w in zip(names, res.weights):
        print(f"  {n:8s} {w:8.4%}")
    print(f"  수익 {res.ret:.4%}  변동성 {res.volatility:.4%}")

    print("\n[2] 목표수익 5% 제약 하 최소분산")
    res = min_variance(mu, cov, target_return=0.05, bounds=(0.0, 1.0))
    for n, w in zip(names, res.weights):
        print(f"  {n:8s} {w:8.4%}")
    print(f"  수익 {res.ret:.4%}  변동성 {res.volatility:.4%}")

    print("\n[3] 효율적 프론티어 (목표수익 스캔)")
    pts = efficient_frontier(mu, cov, n_points=8, bounds=(0.0, 1.0))
    print("   target   ret       vol")
    for p in pts:
        print(f"  {p.target_return:7.3%} {p.ret:7.3%} {p.volatility:7.3%}")

    print("\n[4] L2 정규화 (γ↑ → 분산투자)")
    target = 0.05
    for g in [0.0, 0.005, 0.02, 0.1]:
        r = min_variance_l2(mu, cov, g, target_return=target, bounds=(0.0, 1.0))
        conc = r.weights @ r.weights
        print(f"  γ={g:<6}  집중도 ‖w‖²={conc:.3f}  vol={r.volatility:.4%}  "
              f"w={np.round(r.weights, 3).tolist()}")

    print("\n[5] 특정 자산 배제 (exclude, 이름으로)")
    ex = indices_of(names, ["해외신흥주식"])
    res = min_variance(mu, cov, target_return=0.05, bounds=(0.0, 1.0), exclude=ex)
    for n, w in zip(names, res.weights):
        print(f"  {n:10s} {w:8.4%}")
    print(f"  → 해외신흥주식 배제, 합={res.weights.sum():.4f}")

    print("\n[6] Michaud 리샘플링 (n_sim=50)")
    stdev = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stdev, stdev)
    rs = resample_michaud(mu, stdev, corr, n_sim=50, n_samples=100, seed=1,
                          bounds=(0.0, 1.0), target_return=0.05)
    for n, w in zip(names, rs.weights):
        print(f"  {n:8s} {w:8.4%}")
    print(f"  {rs.message}: ret={rs.ret:.4%} vol={rs.volatility:.4%}")


if __name__ == "__main__":
    main()
