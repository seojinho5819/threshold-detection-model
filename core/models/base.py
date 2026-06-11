"""이상탐지 모델 공통 인터페이스.

향후 AutoEncoder 등으로 교체 가능하도록 추상화한다.
모든 구현체는 "정상 데이터만으로 학습"하고 "점수가 높을수록 정상"인
anomaly score 를 반환하는 규약을 따른다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AnomalyDetector(ABC):
    """이상탐지기 추상 베이스.

    score(X): 값이 클수록 정상, 작을수록 이상인 실수 점수를 반환한다.
    (sklearn IsolationForest.decision_function 과 동일한 방향)
    """

    #: 산출물 식별용 이름 (registry 에서 사용)
    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """anomaly score (높을수록 정상) 반환."""
        ...
