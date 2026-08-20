"""ef_mvo 소스를 읽어 자기완결형 Colab 노트북(.ipynb)을 생성.

노트북은 셀에서 ef_mvo 패키지를 그대로 재생성(%%writefile)한 뒤 실행하므로,
저장소를 clone 하지 않아도 Colab에서 바로 돌아간다. (numpy/scipy는 Colab 내장)

    python build_colab.py   →  ef_mvo_colab.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ef_mvo_colab.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def writefile_cell(relpath: str) -> dict:
    body = (ROOT / relpath).read_text(encoding="utf-8")
    return code(f"%%writefile {relpath}\n{body}")


cells = [
    md(
        "# 효율적 프론티어(MVO) 최적화 — Colab\n"
        "\n"
        "순수 numpy/scipy 포트폴리오 최적화. **설치 불필요** (Colab 내장).\n"
        "\n"
        "**사용법**: 상단 메뉴 `런타임 → 모두 실행` (Runtime → Run all).\n"
        "아래 셀들이 `ef_mvo` 패키지를 만든 뒤 예제를 실행한다.\n"
        "맨 끝 '내 데이터 입력' 셀에서 μ·σ·상관행렬만 바꾸면 내 값으로 계산된다."
    ),
    md("## 1. 패키지 생성 (셀에서 파일로 기록)"),
    code("import os\nos.makedirs('ef_mvo', exist_ok=True)\nprint('ef_mvo/ 준비 완료')"),
    writefile_cell("ef_mvo/optimizer.py"),
    writefile_cell("ef_mvo/sample_data.py"),
    writefile_cell("ef_mvo/__init__.py"),
    md("## 2. 불러오기 & 샘플 데이터 (6자산, PLACEHOLDER 값)"),
    code(
        "import numpy as np\n"
        "from ef_mvo import (min_variance, max_return, efficient_frontier,\n"
        "                    min_variance_l2, resample_michaud, indices_of)\n"
        "from ef_mvo.sample_data import sample\n"
        "\n"
        "mu, cov, names = sample()\n"
        "for n, m, s in zip(names, mu, np.sqrt(np.diag(cov))):\n"
        "    print(f'{n:10s}  μ={m:6.2%}  σ={s:6.2%}')"
    ),
    md("## 3. 최소분산 · 목표수익 제약"),
    code(
        "r = min_variance(mu, cov, target_return=0.05, bounds=(0, 1))\n"
        "print('목표수익 5% 제약 하 최소분산 (성공:', r.success, ')')\n"
        "for n, w in zip(names, r.weights):\n"
        "    print(f'  {n:10s} {w:8.4%}')\n"
        "print(f'  수익 {r.ret:.4%}  변동성 {r.volatility:.4%}')"
    ),
    md("## 4. 효율적 프론티어 (그래프)"),
    code(
        "import matplotlib.pyplot as plt\n"
        "\n"
        "pts = efficient_frontier(mu, cov, n_points=25, bounds=(0, 1))\n"
        "vol = [p.volatility for p in pts]\n"
        "ret = [p.ret for p in pts]\n"
        "plt.figure(figsize=(6, 4))\n"
        "plt.plot(vol, ret, '-o', ms=4)\n"
        "plt.xlabel('Volatility (risk)')\n"
        "plt.ylabel('Expected return')\n"
        "plt.title('Efficient Frontier')\n"
        "plt.grid(alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 5. L2 정규화 (코너해 완화)\n"
        "\n"
        "상관 높은 자산이 극단으로 쏠리는 걸 `γ` 페널티로 부드럽게 분산시킨다."
    ),
    code(
        "print('γ      집중도(‖w‖²)  변동성   비중')\n"
        "for g in [0.0, 0.005, 0.02, 0.1]:\n"
        "    r = min_variance_l2(mu, cov, g, target_return=0.05, bounds=(0, 1))\n"
        "    print(f'{g:<6} {r.weights@r.weights:8.3f}     {r.volatility:6.3%}  '\n"
        "          f'{np.round(r.weights, 3).tolist()}')"
    ),
    md("## 6. 특정 자산 배제 (exclude, 이름으로)"),
    code(
        "ex = indices_of(names, ['해외신흥주식'])\n"
        "r = min_variance(mu, cov, target_return=0.05, bounds=(0, 1), exclude=ex)\n"
        "for n, w in zip(names, r.weights):\n"
        "    print(f'  {n:10s} {w:8.4%}')\n"
        "print('  → 해외신흥주식 배제, 합=%.4f' % r.weights.sum())"
    ),
    md(
        "## 7. 부등식 제약 — 선진/신흥 코너해 방지\n"
        "\n"
        "인덱스: `0 국내주식·1 국내채권·2 해외선진주식·3 해외신흥주식·4 해외채권·5 현금성`\n"
        "\n"
        "- `linear_ineq`   : `aᵀw ≥ b`\n"
        "- `linear_ineq_le`: `aᵀw ≤ c`  (부호 반전 불필요)"
    ),
    code(
        "DEV, EMG = 2, 3\n"
        "def show(tag, w):\n"
        "    print(f'{tag:30s} 선진={w[DEV]:6.2%} 신흥={w[EMG]:6.2%}')\n"
        "\n"
        "# 무제약(코너 가능)\n"
        "show('제약 없음', min_variance(mu, cov, bounds=(0,1)).weights)\n"
        "\n"
        "# ① 자산별 상·하한: 선진·신흥 각 5~30%\n"
        "b = [(0.0,1.0)]*6; b[DEV]=(0.05,0.30); b[EMG]=(0.05,0.30)\n"
        "show('① 각 5~30% (bounds)', min_variance(mu, cov, bounds=b).weights)\n"
        "\n"
        "# ② 상한(≤): 선진+신흥 ≤ 50%\n"
        "g = np.zeros(6); g[DEV]=1.0; g[EMG]=1.0\n"
        "show('② 선진+신흥 ≤ 50% (le)',\n"
        "     min_variance(mu, cov, bounds=(0,1), linear_ineq_le=[(g, 0.5)]).weights)\n"
        "\n"
        "# ③ 비율 밴드: 0.5×선진 ≤ 신흥 ≤ 선진\n"
        "lo = np.zeros(6); lo[EMG]=1.0; lo[DEV]=-0.5   # 신흥 ≥ 0.5선진\n"
        "hi = np.zeros(6); hi[EMG]=1.0; hi[DEV]=-1.0   # 신흥 ≤ 선진 (le)\n"
        "show('③ 0.5선진 ≤ 신흥 ≤ 선진',\n"
        "     min_variance(mu, cov, bounds=(0,1),\n"
        "                  linear_ineq=[(lo, 0.0)], linear_ineq_le=[(hi, 0.0)]).weights)"
    ),
    md(
        "## 8. ✏️ 내 데이터 입력\n"
        "\n"
        "아래 `MY_MU`·`MY_STD`·`MY_CORR` 만 실제 값으로 바꾸고 이 셀을 실행하면 된다.\n"
        "값이 아직 없는 자산은 `exclude` 로 빼 두면 된다."
    ),
    code(
        "# ── 여기만 수정 ───────────────────────────────────────────────\n"
        "MY_NAMES = ['국내주식', '국내채권', '해외선진주식', '해외신흥주식', '해외채권', '현금성']\n"
        "MY_MU  = [0.064, 0.038, 0.075, 0.085, 0.041, 0.022]   # 기대수익 μ\n"
        "MY_STD = [0.160, 0.030, 0.150, 0.200, 0.055, 0.010]   # 표준편차 σ\n"
        "MY_CORR = [\n"
        "    [1.00, 0.10, 0.55, 0.65, 0.20, 0.00],\n"
        "    [0.10, 1.00, 0.05, 0.10, 0.40, 0.15],\n"
        "    [0.55, 0.05, 1.00, 0.70, 0.25, 0.00],\n"
        "    [0.65, 0.10, 0.70, 1.00, 0.30, 0.00],\n"
        "    [0.20, 0.40, 0.25, 0.30, 1.00, 0.10],\n"
        "    [0.00, 0.15, 0.00, 0.00, 0.10, 1.00],\n"
        "]\n"
        "TARGET_RETURN = 0.05\n"
        "# ─────────────────────────────────────────────────────────────\n"
        "\n"
        "mu2  = np.array(MY_MU, float)\n"
        "std2 = np.array(MY_STD, float)\n"
        "corr2 = np.array(MY_CORR, float)\n"
        "cov2 = np.outer(std2, std2) * corr2\n"
        "assert np.linalg.eigvalsh(cov2).min() > -1e-12, 'Σ가 양의정부호가 아님 — 상관행렬 확인'\n"
        "\n"
        "r = min_variance(mu2, cov2, target_return=TARGET_RETURN, bounds=(0, 1))\n"
        "print('성공:', r.success, ' 수익:%.4f'%r.ret, ' 변동성:%.4f'%r.volatility)\n"
        "for n, w in zip(MY_NAMES, r.weights):\n"
        "    print(f'  {n:10s} {w:8.4%}')"
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"생성: {OUT}  (셀 {len(cells)}개)")
