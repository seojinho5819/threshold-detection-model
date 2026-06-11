"""학습 파이프라인 엔트리포인트.

정상 운행 데이터 파일 -> Feature 생성 -> Isolation Forest 학습 ->
Health Score 보정 -> 산출물(model.pkl, feature_config.yaml) 저장.

CLI:
  python -m src.train \
      --data data/normal.csv \
      --config config/feature_config.yaml \
      --out-dir models
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.bundle import ModelBundle
from core.config import FeatureConfig
from core.features import build_features
from core.health_score import HealthCalibration
from core.models import build_detector


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def train(
    data_path: Path,
    config_path: Path,
    out_dir: Path,
    model_type: str = "isolation_forest",
    n_estimators: int = 200,
    contamination: float = 0.01,
    random_state: int = 42,
) -> ModelBundle:
    config = FeatureConfig.load(config_path)
    print(f"[1/5] 설정 로드: {len(config.features)} features, "
          f"window={config.window_seconds}s")

    df = load_dataframe(data_path)
    print(f"[2/5] 데이터 로드: {data_path} (rows={len(df)})")

    feats = build_features(df, config)
    X = feats[config.feature_names].to_numpy()
    print(f"[3/5] Feature 생성: shape={X.shape}")

    detector = build_detector(
        model_type,
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    detector.fit(X)
    print(f"[4/5] 학습 완료: model_type={model_type}")

    # 학습 데이터 점수 분포로 Health Score 보정
    train_scores = detector.score(X)
    calib = HealthCalibration.from_scores(train_scores)

    bundle = ModelBundle(
        detector=detector,
        feature_names=config.feature_names,
        window_seconds=config.window_seconds,
        calibration=calib,
        model_type=model_type,
        trained_at=datetime.now(timezone.utc).isoformat(),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"
    config_out = out_dir / "feature_config.yaml"
    bundle.save(model_path)
    config.save(config_out)
    print(f"[5/5] 저장 완료: {model_path}, {config_out}")
    print(f"      Health 보정: boundary={calib.boundary:.4f}, "
          f"pos_scale={calib.pos_scale:.4f}, neg_scale={calib.neg_scale:.4f}")
    return bundle


def main() -> None:
    p = argparse.ArgumentParser(description="이상탐지 모델 학습")
    p.add_argument("--data", type=Path, required=True, help="정상 운행 데이터 파일")
    p.add_argument("--config", type=Path, default=Path("config/feature_config.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("models"))
    p.add_argument("--model-type", default="isolation_forest")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--contamination", type=float, default=0.01)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    train(
        data_path=args.data,
        config_path=args.config,
        out_dir=args.out_dir,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
