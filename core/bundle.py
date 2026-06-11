"""학습 산출물(model.pkl) 직렬화/역직렬화.

model.pkl 안에 추론에 필요한 모든 것을 담는다:
  - detector      : 학습된 이상탐지기 (스케일러 포함)
  - feature_names : 추론 시 Feature 순서 고정용
  - window_seconds: 버퍼 크기 결정용
  - calibration   : Health Score 보정 파라미터
  - model_type    : 모델 식별자
  - trained_at    : 학습 시각(메타)

feature_config.yaml 은 사람이 읽고 수정하는 입력이며, 별도 산출물로도 저장한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib

from core.health_score import HealthCalibration
from core.models.base import AnomalyDetector


@dataclass
class ModelBundle:
    detector: AnomalyDetector
    feature_names: list[str]
    window_seconds: int
    calibration: HealthCalibration
    model_type: str
    trained_at: str

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "detector": self.detector,
            "feature_names": self.feature_names,
            "window_seconds": self.window_seconds,
            "calibration": self.calibration.to_dict(),
            "model_type": self.model_type,
            "trained_at": self.trained_at,
            "format_version": 1,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        payload = joblib.load(Path(path))
        return cls(
            detector=payload["detector"],
            feature_names=payload["feature_names"],
            window_seconds=payload["window_seconds"],
            calibration=HealthCalibration.from_dict(payload["calibration"]),
            model_type=payload["model_type"],
            trained_at=payload["trained_at"],
        )
