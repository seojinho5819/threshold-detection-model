"""학습 파이프라인 단위/통합 테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import FeatureConfig
from core.features import build_features, compute_features_from_buffer
from core.health_score import (
    HealthCalibration,
    to_health_score,
    to_status,
)
from core.models import build_detector
from training.data import generate

CONFIG_DICT = {
    "window_seconds": 5,
    "sample_rate_hz": 10,
    "time_column": "timestamp",
    "raw_signals": [
        "rpm", "speed", "battery_voltage", "motor_current",
        "motor_temperature", "hydraulic_pressure", "error_count",
    ],
    "features": [
        {"name": "rpm_mean", "source": "rpm", "agg": "mean"},
        {"name": "current_std", "source": "motor_current", "agg": "std"},
        {"name": "temp_max", "source": "motor_temperature", "agg": "max"},
        {"name": "error_sum", "source": "error_count", "agg": "sum"},
    ],
}


@pytest.fixture
def config() -> FeatureConfig:
    return FeatureConfig.from_dict(CONFIG_DICT)


def test_config_validation_rejects_bad_agg():
    bad = dict(CONFIG_DICT)
    bad["features"] = [{"name": "x", "source": "rpm", "agg": "median"}]
    with pytest.raises(ValueError):
        FeatureConfig.from_dict(bad)


def test_config_validation_rejects_unknown_source():
    bad = dict(CONFIG_DICT)
    bad["features"] = [{"name": "x", "source": "unknown", "agg": "mean"}]
    with pytest.raises(ValueError):
        FeatureConfig.from_dict(bad)


def test_build_features_shape_and_names(config):
    df = generate(minutes=1, sample_rate_hz=10, seed=1)
    feats = build_features(df, config)
    assert list(feats.columns) == config.feature_names
    assert len(feats) == len(df)
    assert not feats.isna().any().any()  # NaN 은 0 으로 채워져야 함


def test_buffer_features_match_definition(config):
    df = generate(minutes=1, sample_rate_hz=10, seed=2)
    # 마지막 5초 버퍼를 직접 슬라이싱
    ts = pd.to_datetime(df["timestamp"])
    last = ts.iloc[-1]
    buf = df[ts >= last - pd.Timedelta(seconds=config.window_seconds)]
    row = compute_features_from_buffer(buf, config)
    assert set(row.keys()) == set(config.feature_names)
    # rpm_mean 이 버퍼 rpm 평균과 일치
    assert row["rpm_mean"] == pytest.approx(buf["rpm"].mean())


def test_empty_buffer_returns_zeros(config):
    empty = pd.DataFrame(columns=CONFIG_DICT["raw_signals"] + ["timestamp"])
    row = compute_features_from_buffer(empty, config)
    assert all(v == 0.0 for v in row.values())


def test_health_score_monotonic():
    # boundary=0, 경계 위/아래 스케일 0.2
    calib = HealthCalibration(boundary=0.0, pos_scale=0.2, neg_scale=0.2)
    assert to_health_score(-0.5, calib) == 0       # 매우 이상
    assert to_health_score(0.5, calib) == 100      # 매우 정상
    assert to_health_score(0.0, calib) == 90       # 판정 경계 == normal/warning 경계
    assert 90 < to_health_score(0.1, calib) < 100  # 정상 영역
    assert to_health_score(-0.1, calib) < 90       # 경계 아래


def test_status_thresholds():
    assert to_status(95) == "normal"
    assert to_status(80) == "warning"
    assert to_status(50) == "critical"


def test_train_and_detect_anomalies(config):
    """정상 데이터로 학습 후, 이상 데이터에서 점수가 더 낮아야 한다."""
    normal = generate(minutes=3, sample_rate_hz=10, seed=10)
    anom = generate(minutes=3, sample_rate_hz=10, with_anomalies=True, seed=11)

    X_train = build_features(normal, config)[config.feature_names].to_numpy()
    detector = build_detector("isolation_forest", n_estimators=100)
    detector.fit(X_train)

    calib = HealthCalibration.from_scores(detector.score(X_train))

    X_anom = build_features(anom, config)[config.feature_names].to_numpy()
    anom_scores = detector.score(X_anom)

    # 이상 데이터의 최저 health 가 위험 수준까지 떨어져야 함
    healths = np.array([to_health_score(s, calib) for s in anom_scores])
    assert healths.min() < 70, f"이상 구간 미탐지: min health={healths.min()}"
