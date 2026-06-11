"""anomaly score -> 0~100 Health Score 변환 및 상태 판정.

Backend 와 학습 파이프라인이 동일 로직을 공유한다.

[보정 설계]
이상탐지 모델의 "판정 경계(boundary)"를 기준으로 앵커링한다.
IsolationForest.decision_function 은 boundary=0 을 기준으로
  score >= 0  -> 모델이 정상으로 판정 (학습 데이터의 약 99%)
  score <  0  -> 모델이 이상으로 판정
이 경계가 Health Score 의 normal/warning 경계(기본 90)에 대응되도록 한다.

  - score >= boundary + pos_scale  -> health 100
  - score == boundary              -> health BOUNDARY_HEALTH(=90)
  - score <= boundary - neg_scale  -> health 0
  - 그 사이는 선형 보간

이렇게 하면 정상 데이터는 대부분 90~100(normal)에 모이고,
모델이 이상으로 본 구간만 warning/critical 로 내려간다.
정상 데이터 내부의 미세한 점수 편차가 health 를 깎지 않는다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# 상태 판정 임계값 (Health Score 기준). 설계 문서의 구간과 동일.
NORMAL_MIN = 90  # 90~100 정상
WARNING_MIN = 70  # 70~89 주의, 그 미만 위험(critical)

# 판정 경계(score==boundary)에 대응시킬 Health Score. normal/warning 경계와 일치.
BOUNDARY_HEALTH = 90


@dataclass
class HealthCalibration:
    """학습 데이터 점수 분포 기반 보정 파라미터.

    boundary  : 모델의 정상/이상 판정 경계 점수 (IsolationForest=0.0)
    pos_scale : 경계 위쪽 스케일. (정상 점수 상위 percentile - boundary)
                score 가 이만큼 경계보다 높으면 health=100.
    neg_scale : 경계 아래쪽 스케일. score 가 이만큼 경계보다 낮으면 health=0.
    """

    boundary: float
    pos_scale: float
    neg_scale: float

    @classmethod
    def from_scores(
        cls,
        scores: np.ndarray,
        boundary: float = 0.0,
        high_pct: float = 99.0,
    ) -> "HealthCalibration":
        high = float(np.percentile(scores, high_pct))
        pos_scale = max(high - boundary, 1e-6)
        # 경계 아래 스케일은 정상 데이터의 분포 폭(pos_scale)을 단위로 사용.
        # "정상 산포만큼 경계 아래로 벗어나면 health 0" 이라는 보수적 기준.
        neg_scale = pos_scale
        return cls(boundary=boundary, pos_scale=pos_scale, neg_scale=neg_scale)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HealthCalibration":
        # 구버전(score_low/score_high) 호환 처리
        if "boundary" not in d and "score_high" in d:
            boundary = 0.0
            pos = max(float(d["score_high"]) - boundary, 1e-6)
            return cls(boundary=boundary, pos_scale=pos, neg_scale=pos)
        return cls(
            boundary=float(d["boundary"]),
            pos_scale=float(d["pos_scale"]),
            neg_scale=float(d["neg_scale"]),
        )


def to_health_score(anomaly_score: float, calib: HealthCalibration) -> int:
    """anomaly score(높을수록 정상) -> 0~100 정수 Health Score."""
    delta = anomaly_score - calib.boundary
    if delta >= 0:
        # 경계 위: BOUNDARY_HEALTH ~ 100
        ratio = min(delta / calib.pos_scale, 1.0)
        health = BOUNDARY_HEALTH + (100 - BOUNDARY_HEALTH) * ratio
    else:
        # 경계 아래: BOUNDARY_HEALTH ~ 0
        ratio = min(-delta / calib.neg_scale, 1.0)
        health = BOUNDARY_HEALTH * (1.0 - ratio)
    return int(round(float(np.clip(health, 0, 100))))


def to_status(health_score: int) -> str:
    if health_score >= NORMAL_MIN:
        return "normal"
    if health_score >= WARNING_MIN:
        return "warning"
    return "critical"


def build_result(robot_id: str, anomaly_score: float, calib: HealthCalibration) -> dict:
    """Backend 출력 규격에 맞는 결과 dict 생성."""
    health = to_health_score(anomaly_score, calib)
    return {
        "robot_id": robot_id,
        "health_score": health,
        "anomaly_score": round(float(anomaly_score), 4),
        "status": to_status(health),
    }
