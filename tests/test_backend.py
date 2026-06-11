"""Backend 통합 테스트 (MQTT 비활성, HTTP ingest 경로).

학습 산출물(models/model.pkl)이 있어야 한다. 없으면 스킵.
MQTT_ENABLED=false 로 브로커 없이 실행한다.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

MODEL_PATH = Path("models/model.pkl")
CONFIG_PATH = Path("models/feature_config.yaml")

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and CONFIG_PATH.exists()),
    reason="학습 산출물(models/model.pkl) 없음 - 먼저 training.train 실행 필요",
)


@pytest.fixture
def client():
    os.environ["MQTT_ENABLED"] = "false"
    from fastapi.testclient import TestClient

    from backend.app import app

    with TestClient(app) as c:  # lifespan 실행 -> 모델 로드
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mqtt_enabled"] is False
    assert body["model_type"] == "isolation_forest"


def test_single_sample_is_warmup(client):
    """버퍼가 비면(샘플 1건) 신뢰할 점수가 없으므로 warmup 이어야 한다."""
    sample = {
        "timestamp": "2026-01-01T00:00:00",
        "rpm": 1500, "speed": 8, "battery_voltage": 48,
        "motor_current": 30, "motor_temperature": 55,
        "hydraulic_pressure": 120, "error_count": 0,
    }
    r = client.post("/ingest/robot-cold", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "warmup"
    assert body["health_score"] is None


@pytest.fixture
def service():
    """HTTP 오버헤드 없이 추론 로직을 직접 검증하기 위한 서비스 인스턴스."""
    from backend.inference import InferenceService
    from core.bundle import ModelBundle
    from core.config import FeatureConfig

    bundle = ModelBundle.load(MODEL_PATH)
    config = FeatureConfig.load(CONFIG_PATH)
    return InferenceService(bundle, config)


def _stream_service(service, robot_id, csv_path, max_rows=4000):
    """CSV 를 네이티브 주기(다운샘플 없음)로 ingest. 학습 밀도와 일치.

    워밍업 이후 health 목록 반환.
    """
    df = pd.read_csv(csv_path).iloc[:max_rows]
    healths = []
    for _, row in df.iterrows():
        sample = row.to_dict()
        sample["timestamp"] = str(sample["timestamp"])
        result = service.ingest(robot_id, sample)
        if result["status"] != "warmup":
            healths.append(result["health_score"])
    return healths


def test_normal_stream_stays_healthy(service):
    """정상 스트림(네이티브 주기)은 워밍업 이후 대부분 normal(>=90) 이어야 한다."""
    healths = _stream_service(service, "robot-normal", "data/normal.csv")
    assert healths, "워밍업만 발생, 점수 없음"
    normal_ratio = sum(h >= 90 for h in healths) / len(healths)
    assert normal_ratio > 0.9, f"정상인데 normal 비율 낮음: {normal_ratio:.2f}"


def test_anomaly_stream_drops_health(service):
    """이상 데이터 스트림은 워밍업 이후 최저 health 가 위험 수준까지 내려간다."""
    # 첫 이상 구간(과열, 전체의 20% 지점=1200행)을 포함하도록 충분히 스트리밍
    healths = _stream_service(
        service, "robot-stream", "data/test_with_anomalies.csv", max_rows=2000
    )
    assert healths, "워밍업만 발생, 점수 없음"
    assert min(healths) < 70, f"이상 미탐지: min health={min(healths)}"


def test_listed_after_http_ingest(client):
    """HTTP ingest 후 /robots 목록에 로봇이 등록된다."""
    sample = {
        "timestamp": "2026-01-01T00:00:00",
        "rpm": 1500, "speed": 8, "battery_voltage": 48,
        "motor_current": 30, "motor_temperature": 55,
        "hydraulic_pressure": 120, "error_count": 0,
    }
    client.post("/ingest/robot-listed", json=sample)
    listed = client.get("/robots").json()
    assert any(x["robot_id"] == "robot-listed" for x in listed)


def test_unknown_robot_404(client):
    r = client.get("/robots/does-not-exist")
    assert r.status_code == 404
