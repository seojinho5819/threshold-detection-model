"""로봇별 최근 N초 슬라이딩 버퍼.

MQTT/HTTP 로 들어온 원본 CAN 샘플을 로봇별로 보관하고,
최근 window_seconds 구간을 DataFrame 으로 잘라 Feature 생성에 넘긴다.
스레드(paho 루프)와 요청 핸들러가 동시 접근하므로 Lock 으로 보호한다.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque

import pandas as pd


class BufferStore:
    def __init__(self, window_seconds: int, time_column: str = "timestamp") -> None:
        self.window_seconds = window_seconds
        self.time_column = time_column
        self._lock = threading.Lock()
        # robot_id -> deque[(pd.Timestamp, dict)]
        self._buffers: dict[str, deque] = defaultdict(deque)

    def add(self, robot_id: str, ts: pd.Timestamp, payload: dict) -> None:
        cutoff = ts - pd.Timedelta(seconds=self.window_seconds)
        with self._lock:
            buf = self._buffers[robot_id]
            buf.append((ts, payload))
            # 윈도우를 벗어난 오래된 샘플 제거 (앞에서부터)
            while buf and buf[0][0] < cutoff:
                buf.popleft()

    def window_df(self, robot_id: str) -> pd.DataFrame:
        """해당 로봇의 현재 버퍼를 DataFrame 으로 반환 (시간순)."""
        df, _ = self.snapshot(robot_id)
        return df

    def snapshot(self, robot_id: str) -> tuple[pd.DataFrame, float]:
        """현재 버퍼의 (DataFrame, 시간 span[초]) 를 원자적으로 반환.

        span = 버퍼 내 가장 오래된 샘플과 최신 샘플의 시간 차(초). 샘플<2면 0.
        """
        with self._lock:
            buf = self._buffers.get(robot_id)
            if not buf:
                return pd.DataFrame(), 0.0
            rows = [p for _, p in buf]
            span = (buf[-1][0] - buf[0][0]).total_seconds()
        return pd.DataFrame(rows), float(span)

    def robot_ids(self) -> list[str]:
        with self._lock:
            return list(self._buffers.keys())
