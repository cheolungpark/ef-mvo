"""데모·테스트용 샘플 입력 데이터 (6자산).

⚠️ 아래 μ·σ·상관행렬 값은 전부 **PLACEHOLDER(임시)** 다.
   실제 확정값을 받으면 이 파일의 숫자만 교체하면 된다 — 구조·API는 그대로.

자산 6종: 국내주식·국내채권·해외선진주식·해외신흥주식·해외채권·현금성
  (기존 '해외주식'을 선진/신흥으로 세분화)
Σ 는 σ 와 상관행렬로부터 Σᵢⱼ = σᵢσⱼρᵢⱼ 로 구성해 대칭·양의정부호를 보장한다.
"""

from __future__ import annotations

import numpy as np

ASSET_NAMES = ["국내주식", "국내채권", "해외선진주식", "해외신흥주식", "해외채권", "현금성"]

# --- PLACEHOLDER 값 (실제 값으로 교체 예정) --------------------------------
# 연평균 기대수익률 μ
SAMPLE_MU = np.array([0.064, 0.038, 0.075, 0.085, 0.041, 0.022])  # PLACEHOLDER

# 연 표준편차 σ
SAMPLE_STD = np.array([0.160, 0.030, 0.150, 0.200, 0.055, 0.010])  # PLACEHOLDER

# 상관행렬 ρ (대칭, 대각=1)
SAMPLE_CORR = np.array([  # PLACEHOLDER
    #  국주   국채   선진   신흥   해채   현금
    [1.00,  0.10,  0.55,  0.65,  0.20,  0.00],  # 국내주식
    [0.10,  1.00,  0.05,  0.10,  0.40,  0.15],  # 국내채권
    [0.55,  0.05,  1.00,  0.70,  0.25,  0.00],  # 해외선진주식
    [0.65,  0.10,  0.70,  1.00,  0.30,  0.00],  # 해외신흥주식
    [0.20,  0.40,  0.25,  0.30,  1.00,  0.10],  # 해외채권
    [0.00,  0.15,  0.00,  0.00,  0.10,  1.00],  # 현금성
])

# 공분산행렬 Σ = diag(σ) · ρ · diag(σ)
SAMPLE_COV = np.outer(SAMPLE_STD, SAMPLE_STD) * SAMPLE_CORR


def _validate() -> None:
    """차원 일관성 + 대칭 + 양의정부호(PSD) 검증."""
    n = len(ASSET_NAMES)
    assert SAMPLE_MU.shape == (n,), "μ 길이 불일치"
    assert SAMPLE_STD.shape == (n,), "σ 길이 불일치"
    assert SAMPLE_CORR.shape == (n, n), "상관행렬 크기 불일치"
    assert np.allclose(SAMPLE_CORR, SAMPLE_CORR.T), "상관행렬 비대칭"
    assert np.allclose(np.diag(SAMPLE_CORR), 1.0), "상관행렬 대각 ≠ 1"
    eig = np.linalg.eigvalsh(SAMPLE_COV)
    assert eig.min() > -1e-12, f"Σ가 양의정부호가 아님 (min eig={eig.min():.2e})"


_validate()


def sample() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(mu, cov, asset_names) 복사본을 돌려준다."""
    return SAMPLE_MU.copy(), SAMPLE_COV.copy(), list(ASSET_NAMES)
