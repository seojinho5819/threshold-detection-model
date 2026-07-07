"""데일리 누적 재학습 파이프라인 테스트.

- gate(): 순수 함수라 학습 없이 단위 검증 (부트스트랩/통과/거부)
- collect_recent(): rolling window 파일 선별 검증
- run(): 합성 데이터로 부트스트랩 승격 + 재실행 통합 검증 (학습 산출물 필요 없음)
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from training.daily_retrain import collect_recent, gate, run

DATA_NORMAL = Path("data/normal.csv")
DATA_ANOM = Path("data/test_with_anomalies.csv")
CONFIG = Path("config/feature_config.yaml")

pytestmark = pytest.mark.skipif(
    not (DATA_NORMAL.exists() and DATA_ANOM.exists() and CONFIG.exists()),
    reason="합성 데이터/설정 없음 - generate_synthetic 먼저 실행 필요",
)


# --- gate(): 순수 함수 단위 테스트 ---

def test_gate_bootstrap_passes_without_current():
    ok, reasons = gate({"normal_ratio": 0.5, "detect_ratio": 0.5}, None, 0.01)
    assert ok and "bootstrap" in reasons[0]


def test_gate_passes_when_not_worse():
    cur = {"normal_ratio": 0.90, "detect_ratio": 0.40}
    cand = {"normal_ratio": 0.91, "detect_ratio": 0.40}  # 동등/개선
    ok, _ = gate(cand, cur, 0.01)
    assert ok


def test_gate_rejects_on_normal_ratio_regression():
    cur = {"normal_ratio": 0.95, "detect_ratio": 0.40}
    cand = {"normal_ratio": 0.80, "detect_ratio": 0.40}  # 오탐 급증
    ok, reasons = gate(cand, cur, 0.01)
    assert not ok and any("normal_ratio" in r for r in reasons)


def test_gate_rejects_on_detect_ratio_regression():
    cur = {"normal_ratio": 0.95, "detect_ratio": 0.40}
    cand = {"normal_ratio": 0.95, "detect_ratio": 0.10}  # 미탐 급증
    ok, reasons = gate(cand, cur, 0.01)
    assert not ok and any("detect_ratio" in r for r in reasons)


# --- collect_recent(): 윈도우 선별 ---

def test_collect_recent_picks_window(tmp_path):
    df = pd.read_csv(DATA_NORMAL).iloc[:100]
    for d in ["2026-06-01", "2026-06-09", "2026-06-10", "2026-06-11"]:
        df.to_csv(tmp_path / f"{d}.csv", index=False)
    (tmp_path / "notadate.csv").write_text("x\n1\n", encoding="utf-8")  # 무시돼야 함

    out, files = collect_recent(tmp_path, window_days=3, end_date=date(2026, 6, 11))
    names = sorted(p.stem for p in files)
    # 윈도우 [06-09, 06-11] -> 09,10,11 만 (06-01, notadate 제외)
    assert names == ["2026-06-09", "2026-06-10", "2026-06-11"]
    assert len(out) == 300


# --- run(): 통합 (실제 학습 수행) ---

@pytest.fixture
def daily_dir(tmp_path):
    """학습이 너무 느리지 않게 normal.csv 앞부분을 이틀치 파일로 배치."""
    df = pd.read_csv(DATA_NORMAL).iloc[:5000]
    d = tmp_path / "daily"
    d.mkdir()
    df.iloc[:2500].to_csv(d / "2026-06-10.csv", index=False)
    df.iloc[2500:].to_csv(d / "2026-06-11.csv", index=False)
    return d


def test_run_bootstrap_promotes(tmp_path, daily_dir):
    models = tmp_path / "models"
    result = run(
        daily_dir=daily_dir,
        eval_normal=DATA_NORMAL,
        eval_anomalies=DATA_ANOM,
        config_path=CONFIG,
        models_dir=models,
        window_days=30,
        end_date=date(2026, 6, 11),
    )
    assert result["decision"] == "promoted"
    assert result["exit"] == 0
    assert result["current_metrics"] is None  # 부트스트랩
    assert (models / "model.pkl").exists()
    assert (models / "feature_config.yaml").exists()
    assert (models / "manifest.json").exists()


def test_run_second_pass_archives_and_promotes(tmp_path, daily_dir):
    """현행 모델이 있는 상태에서 동일 데이터로 재실행 -> 악화 없으니 승격 + 백업 생성."""
    models = tmp_path / "models"
    run(  # 1회차: 부트스트랩
        daily_dir=daily_dir, eval_normal=DATA_NORMAL, eval_anomalies=DATA_ANOM,
        config_path=CONFIG, models_dir=models, end_date=date(2026, 6, 11),
    )
    result = run(  # 2회차: 현행 존재
        daily_dir=daily_dir, eval_normal=DATA_NORMAL, eval_anomalies=DATA_ANOM,
        config_path=CONFIG, models_dir=models, end_date=date(2026, 6, 12),
    )
    assert result["decision"] == "promoted"
    assert result["current_metrics"] is not None  # 현행과 비교됨
    assert (models / "archive" / "2026-06-12" / "model.pkl").exists()


def test_run_errors_on_insufficient_data(tmp_path):
    empty = tmp_path / "daily"
    empty.mkdir()
    result = run(
        daily_dir=empty, eval_normal=DATA_NORMAL, eval_anomalies=DATA_ANOM,
        config_path=CONFIG, models_dir=tmp_path / "models", end_date=date(2026, 6, 11),
    )
    assert result["decision"] == "error"
    assert result["exit"] == 1
