"""Backend 런타임 설정 (환경변수 기반).

추가 의존성 없이 os.environ 으로만 구성한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "models/model.pkl")
    feature_config_path: str = os.getenv("FEATURE_CONFIG_PATH", "models/feature_config.yaml")

    # MQTT
    mqtt_enabled: bool = _bool("MQTT_ENABLED", True)
    mqtt_host: str = os.getenv("MQTT_HOST", "localhost")
    mqtt_port: int = int(os.getenv("MQTT_PORT", "1883"))
    # robots/<robot_id>/can 형태 가정. + 와일드카드로 모든 로봇 구독.
    mqtt_topic: str = os.getenv("MQTT_TOPIC", "robots/+/can")
    mqtt_client_id: str = os.getenv("MQTT_CLIENT_ID", "anomaly-backend")

    # 추론 결과(health) 발행. 웹 백엔드가 robots/<id>/health 를 구독해 이벤트화한다.
    # {robot_id} 는 발행 시 치환된다. QoS1=at-least-once(결과 유실 방지),
    # retain=True 면 늦게/재기동 후 구독한 소비자도 최신 health 를 즉시 받는다.
    mqtt_health_topic: str = os.getenv("MQTT_HEALTH_TOPIC", "robots/{robot_id}/health")
    mqtt_health_qos: int = int(os.getenv("MQTT_HEALTH_QOS", "1"))
    mqtt_health_retain: bool = _bool("MQTT_HEALTH_RETAIN", True)

    # 워밍업: 버퍼가 이 시간(초)만큼 차기 전엔 점수 대신 warmup 반환.
    # 빈 문자열/미설정이면 None -> 서비스가 윈도우의 절반으로 자동 결정.
    warmup_min_span_seconds: float | None = (
        float(os.environ["WARMUP_MIN_SPAN_SECONDS"])
        if os.getenv("WARMUP_MIN_SPAN_SECONDS")
        else None
    )

    def __post_init__(self) -> None:
        # 재평가 (dataclass 기본값은 import 시점 1회 평가되므로 안전하게 갱신)
        self.mqtt_enabled = _bool("MQTT_ENABLED", self.mqtt_enabled)
