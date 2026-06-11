"""실시간 추론 서비스.

MQTT 수신 콜백과 HTTP 수신 엔드포인트가 공통으로 사용하는 단일 진입점.
샘플 1건 ingest -> 버퍼 갱신 -> Feature 생성 -> 모델 추론 -> Health Score.

core 의 build_features 와 동일 정의를 공유하는 compute_features_from_buffer 및
build_result 를 그대로 사용해, 학습/추론 간 Feature 불일치를 원천 차단한다.
"""
from __future__ import annotations

import threading

import pandas as pd

from core.bundle import ModelBundle
from core.config import FeatureConfig
from core.features import compute_features_from_buffer
from core.health_score import build_result

from .buffer import BufferStore


class InferenceService:
    def __init__(
        self,
        bundle: ModelBundle,
        config: FeatureConfig,
        warmup_min_span_seconds: float | None = None,
    ) -> None:
        self.bundle = bundle
        self.config = config
        self.buffers = BufferStore(
            window_seconds=bundle.window_seconds,
            time_column=config.time_column,
        )
        # 버퍼가 이만큼(초)의 시간 구간을 채우기 전에는 점수를 신뢰하지 않는다.
        # 빈/희박한 윈도우는 std=0 등으로 학습 분포와 어긋나 오탐을 낸다(콜드스타트).
        # 기본값: 윈도우의 절반. 전송 주기와 무관하게 동작한다.
        self.warmup_min_span = (
            warmup_min_span_seconds
            if warmup_min_span_seconds is not None
            else bundle.window_seconds * 0.5
        )
        self._results_lock = threading.Lock()
        self._latest: dict[str, dict] = {}

    def ingest(self, robot_id: str, sample: dict) -> dict:
        """원본 CAN 샘플 1건 처리 -> 최신 결과 dict 반환."""
        ts = self._parse_ts(sample)
        self.buffers.add(robot_id, ts, sample)

        window, span = self.buffers.snapshot(robot_id)
        n = int(len(window))

        if n < 2 or span < self.warmup_min_span:
            # 워밍업: 아직 신뢰할 점수를 낼 수 없음
            result = {
                "robot_id": robot_id,
                "health_score": None,
                "anomaly_score": None,
                "status": "warmup",
            }
        else:
            feats = compute_features_from_buffer(window, self.config)
            # 학습 시 Feature 순서를 그대로 유지
            x = [[feats[name] for name in self.bundle.feature_names]]
            score = float(self.bundle.detector.score(x)[0])
            result = build_result(robot_id, score, self.bundle.calibration)

        result["timestamp"] = ts.isoformat()
        result["samples_in_window"] = n
        result["window_span_seconds"] = round(span, 2)

        with self._results_lock:
            self._latest[robot_id] = result
        return result

    def get_latest(self, robot_id: str) -> dict | None:
        with self._results_lock:
            return self._latest.get(robot_id)

    def get_all(self) -> list[dict]:
        with self._results_lock:
            return list(self._latest.values())

    def _parse_ts(self, sample: dict) -> pd.Timestamp:
        tcol = self.config.time_column
        raw = sample.get(tcol)
        if raw is None:
            # Agent 가 timestamp 를 안 보내면 수신 시각 사용
            return pd.Timestamp.utcnow()
        return pd.Timestamp(raw)
