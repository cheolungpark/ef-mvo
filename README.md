# ef_mvo — 효율적 프론티어(평균-분산) 최적화

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cheolungpark/ef-mvo/blob/main/ef_mvo_colab.ipynb)

순수 파이썬(numpy/scipy)으로 구현한 표준 포트폴리오 최적화 라이브러리.
Markowitz 평균-분산 최적화, 효율적 프론티어, L2(릿지) 정규화, Michaud 리샘플링을
제공한다.

## 기능

| 함수 | 설명 |
|---|---|
| `min_variance(mu, cov, ...)` | 제약 하 최소분산 포트폴리오 `min wᵀΣw` |
| `max_return(mu, cov, ...)` | 제약 하 기대수익 최대화 (프론티어 상단 경계) |
| `efficient_frontier(mu, cov, ...)` | 목표수익 스캔으로 효율적 프론티어 산출 |
| `min_variance_l2(mu, cov, γ, ...)` | L2 정규화 `min wᵀΣw + γ·wᵀw` |
| `efficient_frontier_l2(mu, cov, γ, ...)` | L2 정규화 프론티어 |
| `fit_l2_gamma(...)` | 목표 비중을 내는 L2 계수 γ 역산 |
| `resample_michaud(mu, stdev, corr, ...)` | Michaud 리샘플링(추정오차 완화) |
| `portfolio_return/variance/volatility` | 포트폴리오 통계 헬퍼 |

제약: 예산 `Σwᵢ=budget`, 비중 상·하한(롱온리 등), 목표수익 `wᵀμ ≥ r*`(또는 등식),
임의의 선형(`linear_ineq`: `aᵀw ≥ b`, `linear_ineq_le`: `aᵀw ≤ c`)·비선형
(`ineq_funcs`: `g(w) ≥ 0`) 제약. 해법은 scipy SLSQP이며, 멀티스타트 +
실행가능성 기반 성공판정으로 SLSQP의 수렴 오탐을 보정한다.

## 구조

```
python/
├─ ef_mvo/
│  ├─ optimizer.py     # 최적화 함수 전체
│  └─ sample_data.py   # 데모·테스트용 샘플 μ·Σ (6자산, PLACEHOLDER)
├─ tests/
│  ├─ conftest.py         # 샘플 mu·cov 픽스처
│  ├─ test_optimizer.py   # 수학적 성질(단위)
│  ├─ test_closed_form.py # 해석적 closed-form 검증
│  └─ test_l2.py          # L2 정규화 성질·closed-form
├─ demo.py
├─ build_colab.py        # 소스 → 자기완결형 Colab 노트북 생성
├─ ef_mvo_colab.ipynb    # Colab 실행용 노트북 (설치·서버 불필요)
├─ requirements.txt
└─ pytest.ini
```

## 웹에서 실행 (Google Colab, 무료·서버 불필요)

`ef_mvo_colab.ipynb` 를 Colab에서 열면 numpy/scipy 내장 환경에서 바로 돌아간다
(설치·서버 비용 0). 노트북이 셀에서 `ef_mvo` 패키지를 직접 만든 뒤 예제·그래프를
실행하고, 마지막 셀에서 `μ·σ·상관행렬` 만 바꾸면 내 값으로 계산된다.

**클릭 한 번 실행**:
👉 https://colab.research.google.com/github/cheolungpark/ef-mvo/blob/main/ef_mvo_colab.ipynb

열린 뒤 `런타임 → 모두 실행`. 맨 끝 셀에서 μ·σ·상관행렬만 바꾸면 내 값으로 계산된다.

노트북을 다시 생성하려면(소스 수정 후): `python build_colab.py`

## 사용

```bash
pip install -r requirements.txt
python demo.py        # 최소분산·프론티어·L2·리샘플링 실행 예시
python -m pytest -q   # 테스트 (19 케이스)
```

```python
import numpy as np
from ef_mvo import min_variance, efficient_frontier, min_variance_l2
from ef_mvo.sample_data import sample

mu, cov, names = sample()

# 목표수익 5% 제약 하 최소분산
res = min_variance(mu, cov, target_return=0.05, bounds=(0.0, 1.0))
print(res.weights, res.ret, res.volatility)

# 효율적 프론티어
for p in efficient_frontier(mu, cov, n_points=25, bounds=(0.0, 1.0)):
    print(p.target_return, p.volatility)

# L2 정규화 (γ↑ → 더 분산된 배분)
res = min_variance_l2(mu, cov, l2_gamma=0.01, target_return=0.05, bounds=(0, 1))
```

### 특정 자산 배제 (`exclude`)

아직 값이 없거나 편입하지 않을 자산을 **비중 0으로 고정**해 배제한다. 출력 비중
벡터의 길이·순서는 유지되어 자산 라벨과 그대로 정렬된다. 값이 확정되면 `exclude`
에서 빼기만 하면 편입된다. 모든 함수(`min_variance`·`efficient_frontier`·
`min_variance_l2`·`resample_michaud`·`fit_l2_gamma`·`max_return`)가 지원한다.

```python
from ef_mvo import indices_of

# 인덱스로
res = min_variance(mu, cov, target_return=0.05, exclude=[3])
# 이름으로 (인덱스·이름 혼용 가능)
res = min_variance(mu, cov, target_return=0.05,
                   exclude=indices_of(names, ["해외신흥주식"]))
```

> `exclude` 는 μ·Σ 에서 행·열을 직접 제거한 (n−1) 결과와 수치적으로 동일하다.
> 데이터가 아직 없는 자산에 NaN·임의값을 넣지 말고 `exclude` 로 빼 두는 것을 권한다.

## L2 정규화

`min wᵀΣw + γ·wᵀw` — L2(릿지) 페널티로 비중을 분산시켜 추정오차에 대한
민감도를 낮춘다. 예산제약 하에서 `γ·wᵀw` 는 `γ·‖w − 1/n‖²`(등가중 편차
페널티)와 동치다. γ→0 이면 순수 최소분산, γ→∞ 이면 등가중에 수렴한다.

## 테스트

- **단위 성질** (`test_optimizer.py`): 예산·롱온리·목표수익 제약 충족, GMV 최소성,
  프론티어 단조성, 리샘플링 시드 재현성 등.
- **해석적 검증** (`test_closed_form.py`): 숏 허용(예산 제약만) 시 최적화 해가
  전역최소분산 `w = Σ⁻¹1/(1ᵀΣ⁻¹1)` 및 프론티어 라그랑주 공식과 일치, 분산의
  볼록성.
- **L2** (`test_l2.py`): γ↑ 시 집중도 `‖w‖²` 감소, γ=0 동치, 대형 γ→등가중,
  `(Σ+γI)⁻¹1` 닫힌 형태 일치, γ 역산 왕복.
