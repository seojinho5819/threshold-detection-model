"""Isolation Forest 기반 이상탐지기.

정상 데이터만으로 학습 가능하고 CPU 에서 경량/실시간 추론이 가능하다.
입력 스케일 영향을 줄이기 위해 StandardScaler 를 파이프라인에 포함한다.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "iforest",
                    IsolationForest(
                        n_estimators=n_estimators,
                        contamination=contamination,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.pipeline.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # decision_function: 양수=정상, 음수=이상 (값이 클수록 정상)
        return self.pipeline.decision_function(X)
