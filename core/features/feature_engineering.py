"""원본 CAN 데이터 -> 최근 N초 통계 Feature 변환.

학습(파일 전체에 대한 슬라이딩 윈도우)과 추론(버퍼 1개)에서
동일한 정의를 공유하도록 한 곳에서 구현한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import FeatureConfig

# config.SUPPORTED_AGGS 와 동기화. pandas Series -> scalar.
AGG_FUNCS = {
    "mean": lambda s: float(s.mean()),
    "std": lambda s: float(s.std(ddof=0)),  # 표본 1개일 때 NaN 대신 0
    "max": lambda s: float(s.max()),
    "min": lambda s: float(s.min()),
    "sum": lambda s: float(s.sum()),
    "last": lambda s: float(s.iloc[-1]),
}

# 슬라이딩 윈도우에서 pandas.rolling 에 넘길 집계 이름 매핑
_ROLLING_AGG = {
    "mean": "mean",
    "std": "std",
    "max": "max",
    "min": "min",
    "sum": "sum",
    "last": "last",
}


def _prepare(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """timestamp 를 datetime 인덱스로 정렬한 사본 반환."""
    tcol = config.time_column
    if tcol not in df.columns:
        raise KeyError(f"time_column '{tcol}' 이 데이터에 없습니다.")
    out = df.copy()
    out[tcol] = pd.to_datetime(out[tcol])
    out = out.set_index(tcol).sort_index()
    return out


def build_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """학습용: 파일 전체에 대해 각 시점 기준 최근 N초 통계 Feature 생성.

    각 행은 "해당 행의 timestamp 에서 끝나는 N초 윈도우" 의 통계다.
    반환 인덱스는 timestamp, 컬럼은 config.feature_names.
    """
    indexed = _prepare(df, config)
    window = f"{config.window_seconds}s"

    out = pd.DataFrame(index=indexed.index)
    # source 별로 rolling 객체를 한 번만 만들어 재사용
    for spec in config.features:
        if spec.source not in indexed.columns:
            raise KeyError(f"source 신호 '{spec.source}' 가 데이터에 없습니다.")
        roller = indexed[spec.source].rolling(window)
        agg_name = _ROLLING_AGG[spec.agg]
        if agg_name == "std":
            series = roller.std(ddof=0)
        elif agg_name == "last":
            series = roller.apply(lambda a: a[-1], raw=True)
        else:
            series = getattr(roller, agg_name)()
        out[spec.name] = series.to_numpy()

    # 윈도우가 1개 표본만 가진 초기 구간의 std=NaN -> 0, 그 외 NaN 도 0 처리
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def compute_features_from_buffer(
    buffer_df: pd.DataFrame, config: FeatureConfig
) -> dict[str, float]:
    """추론용: 최근 N초 버퍼(원본 행들) 1개 -> Feature 1행(dict).

    Backend(MQTT 실시간 수신)에서 매 추론 시 호출하는 경로.
    버퍼 슬라이싱(시간 컷)은 호출 측에서 수행한다고 가정한다.
    """
    if buffer_df.empty:
        # 데이터가 아직 없으면 0 벡터 반환 (추론 측에서 처리)
        return {name: 0.0 for name in config.feature_names}

    row: dict[str, float] = {}
    for spec in config.features:
        if spec.source not in buffer_df.columns:
            raise KeyError(f"source 신호 '{spec.source}' 가 버퍼에 없습니다.")
        series = buffer_df[spec.source].astype(float)
        value = AGG_FUNCS[spec.agg](series)
        row[spec.name] = 0.0 if (value != value) else value  # NaN -> 0
    return row
