"""CSV 파일을 MQTT 로 실시간 publish 하는 시뮬레이터 (실연동 테스트용).

Agent 가 없는 개발 환경에서 robots/<id>/can 토픽에 CAN 샘플을 흘려보낸다.

CLI:
  python -m backend.simulate --data data/test_with_anomalies.csv \
      --robot-id robot-001 --host localhost --speed 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import paho.mqtt.client as mqtt


def main() -> None:
    p = argparse.ArgumentParser(description="CSV -> MQTT 시뮬레이터")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--robot-id", default="robot-001")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--speed", type=float, default=1.0,
                   help="재생 배속 (1.0=실시간, 50=50배속)")
    args = p.parse_args()

    df = (
        pd.read_parquet(args.data)
        if args.data.suffix.lower() == ".parquet"
        else pd.read_csv(args.data)
    )
    topic = f"robots/{args.robot_id}/can"

    client = mqtt.Client(client_id=f"sim-{args.robot_id}")
    client.connect(args.host, args.port, keepalive=60)
    client.loop_start()

    prev_ts = None
    sent = 0
    for _, row in df.iterrows():
        sample = row.to_dict()
        sample["robot_id"] = args.robot_id
        # timestamp 직렬화
        if "timestamp" in sample:
            sample["timestamp"] = str(sample["timestamp"])
        client.publish(topic, json.dumps(sample, default=str))
        sent += 1

        # 원본 샘플 간격을 speed 로 나눠 재생
        ts = pd.Timestamp(row["timestamp"]) if "timestamp" in row else None
        if ts is not None and prev_ts is not None:
            dt = (ts - prev_ts).total_seconds() / max(args.speed, 1e-6)
            if dt > 0:
                time.sleep(min(dt, 1.0))
        prev_ts = ts

    client.loop_stop()
    client.disconnect()
    print(f"전송 완료: {sent} samples -> {topic}")


if __name__ == "__main__":
    main()
