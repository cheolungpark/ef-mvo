"""평균-분산 최적화(MVO) 엔진.

목적함수 ``min wᵀΣw`` (포트폴리오 분산)을 제약 하에서 최소화한다.
  - 변수 : 자산 비중 w
  - 제약 : 예산 Σwᵢ=1, 비중 상·하한(롱온리 등), 목표수익 wᵀμ ≥ r*,
           일반 선형/비선형 제약(자산군 배분밴드 등)
  - 해법 : scipy SLSQP (2차 목적 + 선형·비선형 제약)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------------------
# 결과 컨테이너
# --------------------------------------------------------------------------
@dataclass
class OptResult:
    """단일 포트폴리오 최적화 결과."""

    weights: np.ndarray
    ret: float
    variance: float
    volatility: float
    success: bool
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "ret": self.ret,
            "variance": self.variance,
            "volatility": self.volatility,
            "success": self.success,
            "message": self.message,
        }


@dataclass
class FrontierPoint:
    """효율적 프론티어 위 한 점."""

    target_return: float
    ret: float
    volatility: float
    variance: float
    weights: np.ndarray


# --------------------------------------------------------------------------
# 기본 포트폴리오 통계
# --------------------------------------------------------------------------
def portfolio_return(weights: Sequence[float], mu: Sequence[float]) -> float:
    """기대수익률 μₚ = wᵀμ."""
    return float(np.asarray(weights) @ np.asarray(mu))


def portfolio_variance(weights: Sequence[float], cov: np.ndarray) -> float:
    """분산 σₚ² = wᵀΣw."""
    w = np.asarray(weights, dtype=float)
    return float(w @ np.asarray(cov, dtype=float) @ w)


def portfolio_volatility(weights: Sequence[float], cov: np.ndarray) -> float:
    """변동성 σₚ = √(wᵀΣw)."""
    return float(np.sqrt(portfolio_variance(weights, cov)))


def cov_from_stdev_corr(stdev: Sequence[float], corr: np.ndarray) -> np.ndarray:
    """표준편차 벡터 + 상관행렬 → 공분산행렬  Σᵢⱼ = σᵢσⱼρᵢⱼ."""
    s = np.asarray(stdev, dtype=float)
    return np.outer(s, s) * np.asarray(corr, dtype=float)


# --------------------------------------------------------------------------
# 제약 조립 헬퍼
# --------------------------------------------------------------------------
def _normalize_bounds(bounds, n):
    if bounds is None:
        return [(None, None)] * n
    if isinstance(bounds, tuple) and len(bounds) == 2 and np.isscalar(bounds[0] or 0):
        # 스칼라 튜플 → 전 자산 동일 적용
        return [tuple(bounds)] * n
    bounds = list(bounds)
    if len(bounds) != n:
        raise ValueError(f"bounds 길이({len(bounds)})가 자산 수({n})와 다릅니다.")
    return [tuple(b) for b in bounds]


def _build_constraints(
    n: int,
    mu: np.ndarray,
    budget: float,
    target_return: float | None,
    linear_ineq: Iterable[tuple[Sequence[float], float]] | None,
    ineq_funcs: Iterable[Callable[[np.ndarray], float]] | None,
    equality_return: bool = False,
):
    cons: list[dict] = [
        # 예산제약: Σwᵢ = budget
        {"type": "eq", "fun": lambda w: float(np.sum(w) - budget)},
    ]
    if target_return is not None:
        # equality_return=False → 하한 wᵀμ ≥ r*  (상단 효율적 프론티어)
        # equality_return=True  → 등식 wᵀμ = r*  (프론티어 곡선 전체)
        ctype = "eq" if equality_return else "ineq"
        cons.append(
            {"type": ctype, "fun": lambda w, r=target_return: float(w @ mu - r)}
        )
    for a, b in linear_ineq or []:
        a_vec = np.asarray(a, dtype=float)
        cons.append(
            {"type": "ineq", "fun": lambda w, a=a_vec, b=b: float(a @ w - b)}
        )
    for g in ineq_funcs or []:
        cons.append({"type": "ineq", "fun": g})
    return cons


def indices_of(names: Sequence[str], wanted: Iterable) -> list[int]:
    """자산 이름(또는 정수 인덱스) 목록을 정수 인덱스 목록으로 변환.

    ``exclude=`` 인자에 이름으로 넘길 때 사용:
        exclude=indices_of(names, ["해외신흥주식"])
    정수는 그대로 통과시키므로 이름·인덱스를 섞어 써도 된다.
    """
    out = []
    for item in wanted:
        if isinstance(item, str):
            if item not in names:
                raise KeyError(f"자산 '{item}' 을(를) 찾을 수 없습니다: {list(names)}")
            out.append(list(names).index(item))
        else:
            out.append(int(item))
    return out


def _merge_ineq(linear_ineq, linear_ineq_le):
    """≥ 제약과 ≤ 제약을 하나의 ≥ 목록으로 합친다.  aᵀw ≤ c  ⇔  (−a)ᵀw ≥ −c."""
    ge = list(linear_ineq or [])
    for a, c in (linear_ineq_le or []):
        ge.append((-np.asarray(a, dtype=float), -float(c)))
    return ge


def _apply_exclude(bnds: list, exclude: Iterable[int] | None) -> list:
    """제외할 자산의 상·하한을 (0, 0) 으로 고정 → 비중 0 (사실상 배제)."""
    if not exclude:
        return bnds
    bnds = list(bnds)
    n = len(bnds)
    for i in exclude:
        i = int(i)
        if not -n <= i < n:
            raise IndexError(f"exclude 인덱스 {i} 가 범위(0..{n - 1})를 벗어남")
        bnds[i] = (0.0, 0.0)
    return bnds


def _max_violation(w, cons, bnds, tol=1e-8):
    """제약 위반량의 최댓값(0이면 완전 실행가능)."""
    v = 0.0
    for con in cons:
        val = con["fun"](w)
        if con["type"] == "eq":
            v = max(v, abs(val))
        else:  # ineq: fun ≥ 0
            v = max(v, max(0.0, -val))
    for wi, (lo, hi) in zip(w, bnds):
        if lo is not None:
            v = max(v, max(0.0, lo - wi))
        if hi is not None:
            v = max(v, max(0.0, wi - hi))
    return v


# --------------------------------------------------------------------------
# 최소분산 최적화 (minimize wᵀΣw)
# --------------------------------------------------------------------------
def min_variance(
    mu: Sequence[float],
    cov: np.ndarray,
    *,
    target_return: float | None = None,
    bounds=(0.0, 1.0),
    budget: float = 1.0,
    linear_ineq: Iterable[tuple[Sequence[float], float]] | None = None,
    linear_ineq_le: Iterable[tuple[Sequence[float], float]] | None = None,
    ineq_funcs: Iterable[Callable[[np.ndarray], float]] | None = None,
    l2_gamma: float = 0.0,
    equality_return: bool = False,
    exclude: Iterable[int] | None = None,
    x0: Sequence[float] | None = None,
    ftol: float = 1e-12,
    maxiter: int = 500,
) -> OptResult:
    """제약 하에서 분산을 최소화하는 포트폴리오를 구한다.

    Parameters
    ----------
    mu, cov         기대수익 벡터 μ, 공분산행렬 Σ.
    target_return   설정 시 wᵀμ ≥ target_return 제약 추가(프론티어 스캔용).
    bounds          (lo, hi) 스칼라 튜플이면 전 자산 동일. 자산별 리스트도 허용.
                    롱온리는 lo=0. 자산별 하한을 다르게 줄 수도 있다.
    budget          비중 합(기본 1.0).
    linear_ineq     [(a, b), ...]  → 각 제약 aᵀw ≥ b.
    linear_ineq_le  [(a, c), ...]  → 각 제약 aᵀw ≤ c (내부적으로 −a,−c 로 변환).
                    부호 반전 실수를 줄여주는 ≤ 전용 편의 인자.
    exclude         비중 0으로 고정할 자산 인덱스 목록(사실상 배제). 출력
                    비중 벡터 길이·순서는 유지되어 자산 라벨과 그대로 정렬된다.
                    이름으로 넘기려면 ``exclude=indices_of(names, [...])``.
    linear_ineq     [(a, b), ...]  → 각 제약 aᵀw ≥ b (자산군 배분밴드 등).
    ineq_funcs      [g(w), ...]    → 각 제약 g(w) ≥ 0 (비선형 허용).
    l2_gamma        L2 정규화(릿지) 계수 γ. 목적함수에 γ·wᵀw 를 더해 비중을
                    분산(diversify)시킨다. γ=0 이면 순수 최소분산.
                    (PyPortfolioOpt L2_reg 와 동일 형태. 예산제약 하에서
                    γ·wᵀw 페널티는 γ·‖w−1/n‖² 와 동치.)

    Notes
    -----
    최소화 목적함수:  wᵀΣw + γ·wᵀw
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = mu.shape[0]

    bnds = _apply_exclude(_normalize_bounds(bounds, n), exclude)
    ineq = _merge_ineq(linear_ineq, linear_ineq_le)
    cons = _build_constraints(
        n, mu, budget, target_return, ineq, ineq_funcs, equality_return
    )

    g = float(l2_gamma)
    obj = lambda w: float(w @ cov @ w + g * (w @ w))
    jac = lambda w: 2.0 * cov @ w + 2.0 * g * w

    # 멀티스타트: SLSQP 는 등식 제약에서 'Positive directional derivative'
    # 오탐(실제로는 최적해)을 낼 수 있다. 여러 시작점 중 실행가능하고 목적이
    # 가장 낮은 해를 채택하고, 성공 여부는 제약 위반량으로 재판정한다.
    starts = []
    if x0 is not None:
        starts.append(np.asarray(x0, dtype=float))
    starts.append(np.repeat(budget / n, n))                 # 등가중
    lo = np.array([b[0] if b[0] is not None else 0.0 for b in bnds])
    starts.append(lo + (budget - lo.sum()) / n)             # 하한 + 균등분배
    rng = np.random.default_rng(0)
    for _ in range(3):
        r = rng.random(n)
        starts.append(lo + r / r.sum() * max(budget - lo.sum(), 0.0))

    best = None
    best_msg = ""
    for s in starts:
        res = minimize(obj, s, jac=jac, method="SLSQP", bounds=bnds,
                       constraints=cons, options={"ftol": ftol, "maxiter": maxiter})
        w = res.x
        viol = _max_violation(w, cons, bnds)
        if viol <= 1e-6:
            val = obj(w)
            if best is None or val < best[1] - 1e-12:
                best = (w, val)
                best_msg = str(res.message)
        elif best is None:
            best = (w, obj(w))  # 실행가능 해 없을 때 폴백
            best_msg = f"infeasible (viol={viol:.2e}): {res.message}"

    w, _ = best
    feasible = _max_violation(w, cons, bnds) <= 1e-6
    return OptResult(
        weights=w,
        ret=portfolio_return(w, mu),
        variance=portfolio_variance(w, cov),
        volatility=portfolio_volatility(w, cov),
        success=bool(feasible),
        message=best_msg,
    )


def max_return(
    mu: Sequence[float],
    cov: np.ndarray,
    *,
    bounds=(0.0, 1.0),
    budget: float = 1.0,
    linear_ineq=None,
    linear_ineq_le=None,
    ineq_funcs=None,
    exclude: Iterable[int] | None = None,
    ftol: float = 1e-12,
    maxiter: int = 500,
) -> OptResult:
    """제약 하에서 기대수익을 최대화 (프론티어 상단 경계 산출용)."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = mu.shape[0]
    bnds = _apply_exclude(_normalize_bounds(bounds, n), exclude)
    ineq = _merge_ineq(linear_ineq, linear_ineq_le)
    cons = _build_constraints(n, mu, budget, None, ineq, ineq_funcs)
    res = minimize(
        fun=lambda w: float(-(w @ mu)),
        x0=np.repeat(budget / n, n),
        jac=lambda w: -mu,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": ftol, "maxiter": maxiter},
    )
    w = res.x
    return OptResult(
        weights=w,
        ret=portfolio_return(w, mu),
        variance=portfolio_variance(w, cov),
        volatility=portfolio_volatility(w, cov),
        success=bool(res.success),
        message=str(res.message),
    )


# --------------------------------------------------------------------------
# 효율적 프론티어 (목표수익 스캔)
# --------------------------------------------------------------------------
def efficient_frontier(
    mu: Sequence[float],
    cov: np.ndarray,
    *,
    n_points: int = 25,
    bounds=(0.0, 1.0),
    budget: float = 1.0,
    linear_ineq=None,
    linear_ineq_le=None,
    ineq_funcs=None,
    l2_gamma: float = 0.0,
    exclude: Iterable[int] | None = None,
    ret_range: tuple[float, float] | None = None,
) -> list[FrontierPoint]:
    """효율적 프론티어를 목표수익 스캔으로 산출.

    목표수익을 여러 값으로 바꿔가며 각 지점의 최소분산 포트폴리오를 구한다.
    ``ret_range`` 미지정 시 [전역최소분산 수익, 최대수익] 구간을 균등 분할.
    ``l2_gamma`` > 0 이면 L2 정규화 프론티어를 그린다.
    ``exclude`` 로 특정 자산을 배제할 수 있다(min_variance 참고).
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    common = dict(
        bounds=bounds, budget=budget, linear_ineq=linear_ineq,
        linear_ineq_le=linear_ineq_le, ineq_funcs=ineq_funcs, exclude=exclude,
    )

    if ret_range is None:
        gmv = min_variance(mu, cov, l2_gamma=l2_gamma, **common)
        hi = max_return(mu, cov, **common)
        lo_r, hi_r = gmv.ret, hi.ret
    else:
        lo_r, hi_r = ret_range

    points: list[FrontierPoint] = []
    for t in np.linspace(lo_r, hi_r, n_points):
        r = min_variance(mu, cov, target_return=float(t), l2_gamma=l2_gamma, **common)
        if r.success:
            points.append(
                FrontierPoint(
                    target_return=float(t),
                    ret=r.ret,
                    volatility=r.volatility,
                    variance=r.variance,
                    weights=r.weights,
                )
            )
    return points


# --------------------------------------------------------------------------
# Michaud 리샘플링
# --------------------------------------------------------------------------
def resample_michaud(
    mu: Sequence[float],
    stdev: Sequence[float],
    corr: np.ndarray,
    *,
    n_sim: int,
    n_samples: int = 100,
    seed: int | None = None,
    bounds=(0.0, 1.0),
    budget: float = 1.0,
    target_return: float | None = None,
    linear_ineq=None,
    linear_ineq_le=None,
    ineq_funcs=None,
    exclude: Iterable[int] | None = None,
) -> OptResult:
    """리샘플링 효율(Michaud) 포트폴리오.

    입력 추정치의 불확실성을 몬테카를로로 반영해 안정적인 배분을 얻는다:
      1. 상관행렬 Cholesky 분해로 상관 표준정규 표본 생성
      2. rtn = μ + σ · (chol·z)   (σ = 표준편차)
      3. 표본 평균·공분산 재추정 후 min-variance 재실행
      4. n_sim 회 반복해 나온 비중을 평균
    """
    mu = np.asarray(mu, dtype=float)
    stdev = np.asarray(stdev, dtype=float)
    corr = np.asarray(corr, dtype=float)
    n = mu.shape[0]
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr)  # corr = L Lᵀ

    common = dict(
        target_return=target_return,
        bounds=bounds,
        budget=budget,
        linear_ineq=linear_ineq,
        linear_ineq_le=linear_ineq_le,
        ineq_funcs=ineq_funcs,
        exclude=exclude,
    )

    acc = np.zeros(n)
    ok = 0
    for _ in range(n_sim):
        z = rng.standard_normal((n_samples, n))       # 표준정규
        correlated = z @ L.T                          # 상관 부여
        sim_ret = mu + stdev * correlated             # (n_samples, n)
        sample_mu = sim_ret.mean(axis=0)
        sample_cov = np.cov(sim_ret, rowvar=False)
        r = min_variance(sample_mu, sample_cov, **common)
        if r.success:
            acc += r.weights
            ok += 1
    if ok == 0:
        raise RuntimeError("리샘플링 전 시뮬레이션이 최적화에 실패했습니다.")
    w = acc / ok

    # 원 자산 파라미터(μ, Σ) 기준으로 통계 재계산
    cov = cov_from_stdev_corr(stdev, corr)
    return OptResult(
        weights=w,
        ret=portfolio_return(w, mu),
        variance=portfolio_variance(w, cov),
        volatility=portfolio_volatility(w, cov),
        success=True,
        message=f"resampled over {ok}/{n_sim} sims",
    )


# --------------------------------------------------------------------------
# L2 정규화 편의 래퍼 (EF+L2)
# --------------------------------------------------------------------------
def min_variance_l2(
    mu: Sequence[float],
    cov: np.ndarray,
    l2_gamma: float,
    **kwargs,
) -> OptResult:
    """L2 정규화 최소분산: ``min wᵀΣw + γ·wᵀw``.

    :func:`min_variance` 에 ``l2_gamma`` 를 넘기는 얇은 래퍼.
    γ↑ → 비중이 등가중 쪽으로 퍼져 집중도(‖w‖²)가 낮아진다.
    """
    return min_variance(mu, cov, l2_gamma=l2_gamma, **kwargs)


def efficient_frontier_l2(
    mu: Sequence[float],
    cov: np.ndarray,
    l2_gamma: float,
    **kwargs,
) -> list[FrontierPoint]:
    """L2 정규화 효율적 프론티어(EF+L2)."""
    return efficient_frontier(mu, cov, l2_gamma=l2_gamma, **kwargs)


def fit_l2_gamma(
    mu: Sequence[float],
    cov: np.ndarray,
    target_return: float,
    target_weights: Sequence[float],
    *,
    bounds=(0.0, 1.0),
    budget: float = 1.0,
    linear_ineq=None,
    linear_ineq_le=None,
    ineq_funcs=None,
    exclude: Iterable[int] | None = None,
    bracket=(1e-6, 1.0),
) -> tuple[float, float]:
    """주어진 목표비중을 가장 잘 내는 L2 계수 γ 를 역산.

    ``min_variance_l2(γ)`` 의 해가 ``target_weights`` 에 가장 가까워지는
    γ 를 1차원 최적화로 찾는다. 원하는 분산투자 수준에 맞는 γ 를 고를 때 쓴다.

    Returns
    -------
    (gamma, residual)  residual = 목표 비중과의 최대 절대오차.
    """
    from scipy.optimize import minimize_scalar

    tw = np.asarray(target_weights, dtype=float)
    common = dict(
        target_return=target_return,
        bounds=bounds,
        budget=budget,
        linear_ineq=linear_ineq,
        linear_ineq_le=linear_ineq_le,
        ineq_funcs=ineq_funcs,
        exclude=exclude,
        equality_return=True,  # 특정 프론티어 점 → 수익=타깃 등식
    )

    def loss(log_gamma: float) -> float:
        g = 10.0 ** log_gamma
        w = min_variance(mu, cov, l2_gamma=g, **common).weights
        return float(np.sum((w - tw) ** 2))

    lo, hi = np.log10(bracket[0]), np.log10(bracket[1])
    res = minimize_scalar(loss, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-4})
    gamma = float(10.0 ** res.x)
    w = min_variance(mu, cov, l2_gamma=gamma, **common).weights
    return gamma, float(np.max(np.abs(w - tw)))
