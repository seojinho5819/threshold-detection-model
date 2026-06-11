"""학습된 model.pkl 로 데이터 파일을 추론해 Health Score 를 산출.

Backend 의 실시간 추론 경로를 오프라인 파일에 대해 재현한다.
(Backend 는 동일한 build_features / build_result 를 MQTT 버퍼에 적용)

CLI:
  python -m src.predict --model models/model.pkl \
      --config models/feature_config.yaml \
      --data data/test_with_anomalies.csv --robot-id robot-001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.bundle import ModelBundle
from core.config import FeatureConfig
from core.features import build_features
from core.health_score import build_result


def predict_dataframe(
    df: pd.DataFrame, bundle: ModelBundle, config: FeatureConfig
) -> pd.DataFrame:
    feats = build_features(df, config)
    X = feats[bundle.feature_names].to_numpy()
    scores = bundle.detector.score(X)
    results = [build_result("_", float(s), bundle.calibration) for s in scores]
    out = pd.DataFrame(results, index=feats.index)
    return out[["health_score", "anomaly_score", "status"]]


def main() -> None:
    p = argparse.ArgumentParser(description="model.pkl 추론")
    p.add_argument("--model", type=Path, default=Path("models/model.pkl"))
    p.add_argument("--config", type=Path, default=Path("models/feature_config.yaml"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--robot-id", default="robot-001")
    args = p.parse_args()

    bundle = ModelBundle.load(args.model)
    config = FeatureConfig.load(args.config)
    df = (
        pd.read_parquet(args.data)
        if args.data.suffix.lower() == ".parquet"
        else pd.read_csv(args.data)
    )

    res = predict_dataframe(df, bundle, config)
    counts = res["status"].value_counts().to_dict()
    summary = {
        "robot_id": args.robot_id,
        "rows": int(len(res)),
        "status_counts": counts,
        "health_min": int(res["health_score"].min()),
        "health_mean": round(float(res["health_score"].mean()), 1),
        # 마지막 시점 결과 (실시간 출력 규격 예시)
        "latest": build_result(
            args.robot_id,
            float(bundle.detector.score(
                build_features(df, config)[bundle.feature_names].to_numpy()[-1:],
            )[0]),
            bundle.calibration,
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
