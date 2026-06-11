"""모델 종류 -> 구현체 매핑.

설정/CLI 에서 모델 타입을 문자열로 선택할 수 있게 하고,
향후 AutoEncoder 등을 추가할 때 이 곳에만 등록하면 되도록 한다.
"""
from __future__ import annotations

from typing import Any

from .base import AnomalyDetector
from .isolation_forest import IsolationForestDetector

_REGISTRY: dict[str, type[AnomalyDetector]] = {
    "isolation_forest": IsolationForestDetector,
    # "autoencoder": AutoEncoderDetector,  # 향후 확장 지점
}


def build_detector(model_type: str, **params: Any) -> AnomalyDetector:
    if model_type not in _REGISTRY:
        raise ValueError(
            f"알 수 없는 모델 타입 '{model_type}'. "
            f"가능: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[model_type](**params)
