"""feature_config.yaml 로딩 및 검증.

설정 파일 하나로 Feature 구성을 관리한다는 설계 원칙을 구현한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 지원하는 집계 함수 목록 (feature_engineering 의 AGG_FUNCS 와 동기화)
SUPPORTED_AGGS = {"mean", "std", "max", "min", "sum", "last"}


@dataclass(frozen=True)
class FeatureSpec:
    """통계 Feature 한 개의 정의."""

    name: str
    source: str
    agg: str


@dataclass
class FeatureConfig:
    window_seconds: int
    time_column: str
    raw_signals: list[str]
    features: list[FeatureSpec]
    sample_rate_hz: int = 10
    # 원본 설정 dict 보존 (산출물로 다시 저장할 때 사용)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @classmethod
    def load(cls, path: str | Path) -> "FeatureConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureConfig":
        features = [
            FeatureSpec(name=f["name"], source=f["source"], agg=f["agg"])
            for f in data["features"]
        ]
        cfg = cls(
            window_seconds=int(data["window_seconds"]),
            time_column=data.get("time_column", "timestamp"),
            raw_signals=list(data["raw_signals"]),
            features=features,
            sample_rate_hz=int(data.get("sample_rate_hz", 10)),
            raw=data,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("window_seconds 는 양수여야 합니다.")
        signal_set = set(self.raw_signals)
        for f in self.features:
            if f.agg not in SUPPORTED_AGGS:
                raise ValueError(
                    f"지원하지 않는 집계 함수 '{f.agg}' (feature={f.name}). "
                    f"가능: {sorted(SUPPORTED_AGGS)}"
                )
            if f.source not in signal_set:
                raise ValueError(
                    f"feature '{f.name}' 의 source '{f.source}' 가 "
                    f"raw_signals 에 없습니다."
                )
        names = self.feature_names
        if len(names) != len(set(names)):
            raise ValueError("feature name 이 중복되었습니다.")

    def save(self, path: str | Path) -> None:
        """학습에 사용된 설정을 산출물로 저장 (Backend 배포용)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.raw, fh, allow_unicode=True, sort_keys=False)
