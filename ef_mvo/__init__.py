"""효율적 프론티어(평균-분산) 포트폴리오 최적화 라이브러리.

순수 파이썬(numpy/scipy)으로 구현한 표준 기능:
  - ``min_variance``       : 제약 하 최소분산(Markowitz) 포트폴리오
  - ``efficient_frontier`` : 목표수익 스캔으로 효율적 프론티어 산출
  - ``min_variance_l2``    : L2(릿지) 정규화로 분산투자 유도
  - ``resample_michaud``   : Michaud 리샘플링(추정오차 완화)
"""

from .optimizer import (
    OptResult,
    FrontierPoint,
    cov_from_stdev_corr,
    portfolio_return,
    portfolio_variance,
    portfolio_volatility,
    min_variance,
    max_return,
    efficient_frontier,
    resample_michaud,
    min_variance_l2,
    efficient_frontier_l2,
    fit_l2_gamma,
    indices_of,
)

__all__ = [
    "OptResult",
    "FrontierPoint",
    "cov_from_stdev_corr",
    "portfolio_return",
    "portfolio_variance",
    "portfolio_volatility",
    "min_variance",
    "max_return",
    "efficient_frontier",
    "resample_michaud",
    "min_variance_l2",
    "efficient_frontier_l2",
    "fit_l2_gamma",
    "indices_of",
]
