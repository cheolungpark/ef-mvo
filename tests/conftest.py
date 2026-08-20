import sys
from pathlib import Path

import pytest

# 패키지 루트를 import 경로에 추가 (별도 설치 없이 pytest 실행 가능)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ef_mvo.sample_data import sample  # noqa: E402


@pytest.fixture(scope="session")
def mu():
    m, _, _ = sample()
    return m


@pytest.fixture(scope="session")
def cov():
    _, c, _ = sample()
    return c
