"""ef_mvo 소스를 읽어 '컴맹용 폼 UI' Colab 노트북(.ipynb)을 생성.

노트북은 셀에서 ef_mvo 패키지를 재생성(%%writefile)한 뒤, ipywidgets 폼을
띄운다. 사용자는 표에 숫자만 넣고 '▶ 계산하기'를 누르면 결과 표·그래프가
나온다(코드 편집 불필요). numpy/scipy/ipywidgets 는 Colab 내장.

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


# --- 폰트 설정 + 임포트 -----------------------------------------------------
SETUP = r"""# (준비) 한글 그래프 폰트 + 라이브러리 로드 — 자동 실행, 건드리지 마세요
try:
    import os, matplotlib, matplotlib.font_manager as fm
    if not any('Nanum' in f.name for f in fm.fontManager.ttflist):
        os.system('apt-get -qq install -y fonts-nanum > /dev/null 2>&1')
        for fp in fm.findSystemFonts(fontpaths=['/usr/share/fonts/truetype/nanum']):
            fm.fontManager.addfont(fp)
    matplotlib.rc('font', family='NanumGothic')
    matplotlib.rc('axes', unicode_minus=False)
except Exception as e:
    print('폰트 설정 건너뜀(그래프 한글이 깨질 수 있으나 표는 정상):', e)

import numpy as np
from ef_mvo import min_variance, efficient_frontier
from ef_mvo.sample_data import ASSET_NAMES, SAMPLE_MU, SAMPLE_STD, SAMPLE_CORR
print('준비 완료 ✅  아래 폼에서 입력하세요.')
"""

# --- 순수 계산 함수 (ipywidgets 불필요) -------------------------------------
RUN_OPT = r"""# (준비) 계산 함수 — 자동 실행, 건드리지 마세요
def run_opt(mu, sd, corr, mins, maxs, include, target, gamma, le=None):
    mu = np.asarray(mu, float); sd = np.asarray(sd, float)
    corr = np.asarray(corr, float)
    cov = np.outer(sd, sd) * corr
    if np.linalg.eigvalsh(cov).min() < -1e-8:
        return {'error': '상관행렬 값이 수학적으로 불가능한 조합입니다. '
                         '고급▸상관행렬 값을 확인하세요(보통 -1~1).'}
    n = len(mu)
    exclude = [i for i in range(n) if not include[i]]
    bounds = [(mins[i], maxs[i]) for i in range(n)]
    if sum(maxs[i] for i in range(n) if include[i]) < 0.999:
        return {'error': '포함한 자산들의 최대비중 합이 100%보다 작아 합계 100%가 '
                         '불가능합니다. 최대(%)를 늘리세요.'}
    r = min_variance(mu, cov, target_return=target, bounds=bounds,
                     exclude=exclude, l2_gamma=gamma, linear_ineq_le=le)
    if not r.success:
        return {'error': '조건이 너무 빡빡해 답을 못 찾았어요. 최소/최대 비중을 '
                         '넓히거나 목표수익을 낮춰보세요.'}
    pts = efficient_frontier(mu, cov, n_points=25, bounds=bounds,
                             exclude=exclude, l2_gamma=gamma, linear_ineq_le=le)
    return {'r': r, 'pts': pts}
"""

# --- ipywidgets 폼 ----------------------------------------------------------
FORM = r"""# 입력 폼 — 표에 숫자를 넣고 아래 '▶ 계산하기'를 누르세요
import ipywidgets as W
from IPython.display import display, clear_output, HTML
import matplotlib.pyplot as plt

names = list(ASSET_NAMES); n = len(names)
W_ = lambda px: W.Layout(width=px)

mu_in, sd_in, min_in, max_in, inc_in = [], [], [], [], []
header = W.HBox([W.HTML('<b>자산</b>', layout=W_('120px')),
                W.HTML('<b>기대수익 μ(%)</b>', layout=W_('130px')),
                W.HTML('<b>변동성 σ(%)</b>', layout=W_('130px')),
                W.HTML('<b>최소(%)</b>', layout=W_('90px')),
                W.HTML('<b>최대(%)</b>', layout=W_('90px')),
                W.HTML('<b>포함</b>', layout=W_('55px'))])
rows = []
for i, nm in enumerate(names):
    m = W.FloatText(value=round(float(SAMPLE_MU[i])*100, 2), layout=W_('130px'))
    s = W.FloatText(value=round(float(SAMPLE_STD[i])*100, 2), layout=W_('130px'))
    lo = W.FloatText(value=0.0, layout=W_('90px'))
    hi = W.FloatText(value=100.0, layout=W_('90px'))
    inc = W.Checkbox(value=True, indent=False, layout=W_('55px'))
    mu_in.append(m); sd_in.append(s); min_in.append(lo); max_in.append(hi); inc_in.append(inc)
    rows.append(W.HBox([W.HTML(nm, layout=W_('120px')), m, s, lo, hi, inc]))

# 상관행렬(고급, 기본 접힘)
corr_in = {}
grid = []
for i in range(n):
    cells = [W.HTML(names[i][:5], layout=W_('80px'))]
    for j in range(n):
        if j <= i:
            cells.append(W.HTML('', layout=W_('62px')))
        else:
            t = W.FloatText(value=round(float(SAMPLE_CORR[i][j]), 2), layout=W_('62px'))
            corr_in[(i, j)] = t; cells.append(t)
    grid.append(W.HBox(cells))
adv = W.Accordion(children=[W.VBox([W.HTML('상관계수(-1~1). 잘 모르면 그대로 두세요.')] + grid)])
adv.set_title(0, '⚙️ 고급: 상관행렬 (안 바꿔도 됨)'); adv.selected_index = None

target = W.FloatSlider(value=5.0, min=2.0, max=9.0, step=0.1, description='목표수익(%)',
                       style={'description_width': '110px'}, layout=W_('420px'),
                       readout_format='.1f')
l2 = W.FloatSlider(value=0.0, min=0.0, max=0.05, step=0.005, description='고르게 나누기',
                   style={'description_width': '110px'}, layout=W_('420px'),
                   readout_format='.3f')

# 관계형 제약: 해외신흥주식 ≤ 해외선진주식 × 배수
try:
    IDX_DEV = names.index('해외선진주식'); IDX_EMG = names.index('해외신흥주식')
except ValueError:
    IDX_DEV = IDX_EMG = None
ratio_on = W.Checkbox(value=False, indent=False,
                      description='해외신흥 ≤ 해외선진 × 배수', layout=W_('260px'))
ratio_val = W.BoundedFloatText(value=1.0, min=0.0, max=5.0, step=0.1, layout=W_('80px'))
ratio_box = W.HBox([ratio_on, W.HTML('배수:', layout=W_('40px')), ratio_val])

btn = W.Button(description='▶ 계산하기', button_style='success', layout=W.Layout(width='220px', height='42px'))
out = W.Output()

def on_click(_):
    with out:
        clear_output()
        mu = [x.value/100 for x in mu_in]; sd = [x.value/100 for x in sd_in]
        mins = [x.value/100 for x in min_in]; maxs = [x.value/100 for x in max_in]
        include = [c.value for c in inc_in]
        corr = np.eye(n)
        for (i, j), t in corr_in.items():
            corr[i, j] = corr[j, i] = t.value
        # 관계형 제약: 신흥 ≤ 선진 × 배수  →  w_신흥 - 배수·w_선진 ≤ 0
        le = None
        if ratio_on.value and IDX_DEV is not None:
            a = np.zeros(n); a[IDX_EMG] = 1.0; a[IDX_DEV] = -ratio_val.value
            le = [(a, 0.0)]
        res = run_opt(mu, sd, corr, mins, maxs, include, target.value/100, l2.value, le)
        if 'error' in res:
            display(HTML(f"<b style='color:#c0392b'>⚠️ {res['error']}</b>")); return
        r, pts = res['r'], res['pts']
        h = ("<h3>📊 추천 비중</h3>"
             "<table style='border-collapse:collapse' border='1' cellpadding='6'>"
             "<tr style='background:#eee'><th>자산</th><th>비중</th></tr>")
        for nm, wt in zip(names, r.weights):
            bar = '█' * int(round(wt*25))
            h += f"<tr><td>{nm}</td><td style='font-family:monospace'>{wt*100:6.2f}%  <span style='color:#27ae60'>{bar}</span></td></tr>"
        h += (f"<tr style='background:#f6f6f6'><td><b>합계</b></td>"
              f"<td><b>{r.weights.sum()*100:.1f}%</b></td></tr></table>"
              f"<p>📈 예상 수익률 <b>{r.ret*100:.2f}%</b> &nbsp;·&nbsp; "
              f"📉 예상 위험(변동성) <b>{r.volatility*100:.2f}%</b></p>")
        display(HTML(h))
        plt.figure(figsize=(6, 4))
        plt.plot([p.volatility*100 for p in pts], [p.ret*100 for p in pts], '-o', ms=3,
                 label='효율적 프론티어')
        plt.scatter([r.volatility*100], [r.ret*100], c='red', s=110, zorder=5,
                    label='내 포트폴리오')
        plt.xlabel('위험(변동성) %'); plt.ylabel('기대수익 %')
        plt.title('효율적 프론티어'); plt.legend(); plt.grid(alpha=0.3); plt.show()

btn.on_click(on_click)
display(W.HTML("<h2>① 표에 숫자 입력 → ② '▶ 계산하기' 클릭</h2>"
               "<p>· 어떤 자산을 빼려면 <b>포함</b> 체크 해제<br>"
               "· 한 자산이 너무 쏠리면 <b>최대(%)</b>를 낮추세요(예: 30)<br>"
               "· <b>고르게 나누기</b>를 올리면 비중이 더 분산됩니다<br>"
               "· <b>해외신흥 ≤ 해외선진</b>: 체크하면 신흥 비중이 선진을 넘지 않음"
               "(배수 0.5면 선진의 절반까지)</p>"))
display(header, *rows, adv, target, l2, ratio_box, btn, out)
"""

cells = [
    md(
        "# 📈 포트폴리오 자동 배분 (효율적 프론티어)\n"
        "\n"
        "숫자만 넣으면 **위험 대비 수익이 가장 좋은 자산 비중**을 계산해 줍니다.\n"
        "\n"
        "## 딱 3단계\n"
        "1. 위 메뉴 **`런타임 → 모두 실행`** 클릭 (처음 한 번, 20~30초)\n"
        "2. 맨 아래 **표에 숫자 입력**\n"
        "3. **`▶ 계산하기`** 버튼 클릭 → 결과 표·그래프 확인\n"
        "\n"
        "> 코드는 안 봐도 됩니다. 아래 '준비' 셀들은 자동으로 돌아갑니다."
    ),
    md("---\n### ⬇️ 준비 셀 (자동 실행 · 건드리지 마세요)"),
    code("import os\nos.makedirs('ef_mvo', exist_ok=True)"),
    writefile_cell("ef_mvo/optimizer.py"),
    writefile_cell("ef_mvo/sample_data.py"),
    writefile_cell("ef_mvo/__init__.py"),
    code(SETUP),
    code(RUN_OPT),
    md("---\n# ✏️ 여기서 입력하세요"),
    code(FORM),
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
