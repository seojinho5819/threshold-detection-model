"""FastAPI 애플리케이션.

서버 시작 시 model.pkl + feature_config.yaml 로드 -> InferenceService 생성 ->
(옵션) MQTT 수신 시작. 엔드포인트로 로봇별 Health Score 를 조회한다.

실행:
  uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.bundle import ModelBundle
from core.config import FeatureConfig

from .inference import InferenceService
from .settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend.app")

# 애플리케이션 상태 (lifespan 에서 채움)
state: dict = {"service": None, "mqtt": None, "settings": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    state["settings"] = settings

    logger.info("모델 로드: %s", settings.model_path)
    bundle = ModelBundle.load(settings.model_path)
    config = FeatureConfig.load(settings.feature_config_path)
    service = InferenceService(
        bundle, config, warmup_min_span_seconds=settings.warmup_min_span_seconds
    )
    state["service"] = service
    logger.info(
        "추론 준비 완료: model_type=%s, features=%d, window=%ds",
        bundle.model_type, len(bundle.feature_names), bundle.window_seconds,
    )

    if settings.mqtt_enabled:
        # MQTT 의존성은 활성화된 경우에만 import (테스트/오프라인 실행 편의)
        from .mqtt_client import MqttIngestor

        ingestor = MqttIngestor(service, settings)
        try:
            ingestor.start()
            state["mqtt"] = ingestor
        except Exception as exc:  # noqa: BLE001 - 브로커 미가동 시에도 API 는 떠야 함
            logger.warning("MQTT 시작 실패(브로커 미가동?), API 만 기동: %s", exc)
    else:
        logger.info("MQTT 비활성화 (MQTT_ENABLED=false)")

    yield

    if state.get("mqtt"):
        state["mqtt"].stop()


app = FastAPI(title="AI 이상탐지 Backend", version="0.1.0", lifespan=lifespan)


def _service() -> InferenceService:
    svc = state.get("service")
    if svc is None:
        raise HTTPException(status_code=503, detail="서비스 미준비")
    return svc


class Sample(BaseModel):
    """HTTP 수신용 CAN 샘플 (테스트/직접 연동용). 추가 신호는 자유롭게 허용."""

    model_config = {"extra": "allow"}


@app.get("/health")
def health():
    svc = state.get("service")
    settings: Settings = state.get("settings")
    return {
        "status": "ok" if svc else "starting",
        "model_type": svc.bundle.model_type if svc else None,
        "trained_at": svc.bundle.trained_at if svc else None,
        "mqtt_enabled": settings.mqtt_enabled if settings else None,
        "mqtt_connected": state.get("mqtt") is not None,
        "robots": len(svc.get_all()) if svc else 0,
    }


@app.get("/robots")
def list_robots():
    """모든 로봇의 최신 Health Score (Frontend 장비 목록용)."""
    return _service().get_all()


@app.get("/robots/{robot_id}")
def get_robot(robot_id: str):
    result = _service().get_latest(robot_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"robot '{robot_id}' 데이터 없음")
    return result


@app.post("/ingest/{robot_id}")
def ingest(robot_id: str, sample: Sample):
    """HTTP 로 CAN 샘플 1건 주입 (MQTT 대체/테스트 경로). 결과 즉시 반환."""
    return _service().ingest(robot_id, sample.model_dump())
