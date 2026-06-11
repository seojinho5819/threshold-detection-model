"""MQTT 수신 클라이언트.

Agent 가 robots/<robot_id>/can 토픽으로 publish 하는 CAN 샘플(JSON)을 수신해
InferenceService.ingest 로 흘려보낸다. paho 의 네트워크 루프는 별도 스레드에서
동작하므로(loop_start), 공유 상태는 InferenceService 내부 Lock 으로 보호된다.

페이로드 예: {"robot_id":"robot-001","timestamp":"...","rpm":1500, ...}
robot_id 는 페이로드 우선, 없으면 토픽에서 파싱.
"""
from __future__ import annotations

import json
import logging

import paho.mqtt.client as mqtt

from .inference import InferenceService
from .settings import Settings

logger = logging.getLogger("backend.mqtt")


class MqttIngestor:
    def __init__(self, service: InferenceService, settings: Settings) -> None:
        self.service = service
        self.settings = settings
        self._client = mqtt.Client(client_id=settings.mqtt_client_id, clean_session=True)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def start(self) -> None:
        s = self.settings
        logger.info("MQTT 연결 시도: %s:%s topic=%s", s.mqtt_host, s.mqtt_port, s.mqtt_topic)
        self._client.connect(s.mqtt_host, s.mqtt_port, keepalive=60)
        self._client.loop_start()

    def stop(self) -> None:
        self._client.loop_stop()
        try:
            self._client.disconnect()
        except Exception:  # pragma: no cover - 종료 경로
            pass

    # --- paho 콜백 (네트워크 루프 스레드에서 호출됨) ---
    def _on_connect(self, client, userdata, flags, rc):  # noqa: ANN001
        logger.info("MQTT 연결됨 rc=%s, 구독: %s", rc, self.settings.mqtt_topic)
        client.subscribe(self.settings.mqtt_topic)

    def _on_message(self, client, userdata, msg):  # noqa: ANN001
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            logger.warning("잘못된 페이로드 무시 topic=%s: %s", msg.topic, exc)
            return
        robot_id = payload.get("robot_id") or self._robot_id_from_topic(msg.topic)
        if not robot_id:
            logger.warning("robot_id 없음, 무시 topic=%s", msg.topic)
            return
        try:
            self.service.ingest(robot_id, payload)
        except Exception as exc:  # noqa: BLE001 - 콜백에서 예외가 루프를 죽이지 않게
            logger.exception("추론 실패 robot=%s: %s", robot_id, exc)

    @staticmethod
    def _robot_id_from_topic(topic: str) -> str | None:
        # robots/<robot_id>/can
        parts = topic.split("/")
        if len(parts) >= 2 and parts[0] == "robots":
            return parts[1]
        return None
