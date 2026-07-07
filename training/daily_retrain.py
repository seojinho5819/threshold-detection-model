"""데일리 누적 재학습 오케스트레이터.

흐름:
  1. 수집/누적  : data/daily/ 에서 최근 N일(rolling window) 파일을 concat
  2. 학습       : 누적 데이터로 후보 모델 from-scratch 학습 (candidates/<date>)
  3. 검증(게이트): 고정 eval셋으로 후보 vs 현행 비교
                  - normal_ratio  (정상셋의 normal 비율, 높을수록 오탐 적음)
                  - detect_ratio  (이상셋의 비-normal 비율, 높을수록 미탐 적음)
  4. 승격       : 둘 다 현행 대비 악화 없을 때만 원자적 교체(+백업+manifest)
                  현행 모델이 없으면 부트스트랩으로 무조건 승격
  5. 거부       : 미통과 시 후보만 남기고 현행 유지

Isolation Forest 는 incremental 학습이 불가하므로 "누적 학습"=매번 누적 데이터로
from-scratch 재학습이다. 재학습이 정확도를 올리는 메커니즘은 정상 envelope 의
커버리지 확대(오탐 감소) + 드리프트 적응이다.

종료 코드:
  0  승격됨(또는 부트스트랩)  -> 스케줄러가 이어서 추론 백엔드 재시작
  2  검증 미통과로 거부, 현행 유지 -> 재시작 불필요
  1  학습 데이터 부족 등 오류

CLI 예:
  python -m training.daily_retrain --daily-dir data/daily \
      --eval-normal data/eval/normal.csv --eval-anomalies data/eval/anomalies.csv \
      --window-days 30 --models-dir models
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from core.bundle import ModelBundle
from core.config import FeatureConfig
from training.predict import predict_dataframe
from training.train import train


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def collect_recent(
    daily_dir: str | Path, window_days: int, end_date: date
) -> tuple[pd.DataFrame, list[Path]]:
    """daily_dir 에서 파일명이 YYYY-MM-DD 인 파일 중 최근 window_days 일치를 concat.

    [end_date - (window_days-1), end_date] 구간 파일만 사용. 시간순 정렬.
    """
    daily_dir = Path(daily_dir)
    start = end_date - timedelta(days=window_days - 1)
    picked: list[tuple[date, Path]] = []
    if daily_dir.exists():
        for p in daily_dir.iterdir():
            if p.suffix.lower() not in (".csv", ".parquet"):
                continue
            try:
                d = date.fromisoformat(p.stem)
            except ValueError:
                continue  # 날짜 규약 아닌 파일은 무시
            if start <= d <= end_date:
                picked.append((d, p))
    picked.sort(key=lambda t: t[0])
    files = [p for _, p in picked]
    if not files:
        return pd.DataFrame(), []
    df = pd.concat([_read_table(p) for p in files], ignore_index=True)
    return df, files


def evaluate(
    bundle: ModelBundle,
    config: FeatureConfig,
    eval_normal: str | Path,
    eval_anomalies: str | Path,
) -> dict:
    """고정 eval셋으로 모델을 채점. 라벨 없는 비지도 모델의 상대 비교용 지표."""
    res_n = predict_dataframe(_read_table(eval_normal), bundle, config)
    res_a = predict_dataframe(_read_table(eval_anomalies), bundle, config)
    return {
        # 정상셋에서 normal 로 본 비율 (오탐이 적을수록 ↑)
        "normal_ratio": round(float((res_n["status"] == "normal").mean()), 4),
        # 이상셋에서 비-normal(warning/critical) 로 본 비율 (미탐이 적을수록 ↑)
        "detect_ratio": round(float((res_a["status"] != "normal").mean()), 4),
    }


def gate(candidate: dict, current: dict | None, tol: float) -> tuple[bool, list[str]]:
    """승격 게이트. 현행이 없으면 부트스트랩 통과. 있으면 두 지표 모두 악화 없을 때만."""
    if current is None:
        return True, ["bootstrap: 현행 모델 없음 -> 무조건 승격"]
    reasons: list[str] = []
    ok = True
    if candidate["normal_ratio"] < current["normal_ratio"] - tol:
        ok = False
        reasons.append(
            f"normal_ratio 악화: 후보 {candidate['normal_ratio']} "
            f"< 현행 {current['normal_ratio']} - {tol}"
        )
    if candidate["detect_ratio"] < current["detect_ratio"] - tol:
        ok = False
        reasons.append(
            f"detect_ratio 악화: 후보 {candidate['detect_ratio']} "
            f"< 현행 {current['detect_ratio']} - {tol}"
        )
    if ok:
        reasons.append("후보가 현행 대비 악화 없음")
    return ok, reasons


def _atomic_replace(src: Path, dst: Path) -> None:
    """같은 파일시스템에서 임시본 경유로 원자적 교체."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)  # 같은 fs 내 atomic


def _promote(cand_model: Path, cand_cfg: Path, models_dir: Path, end_date: date) -> None:
    """현행 모델을 archive 로 백업하고 후보를 production 위치로 원자적 교체."""
    cur_model = models_dir / "model.pkl"
    cur_cfg = models_dir / "feature_config.yaml"
    if cur_model.exists():
        arch = models_dir / "archive" / end_date.isoformat()
        arch.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cur_model, arch / "model.pkl")
        if cur_cfg.exists():
            shutil.copy2(cur_cfg, arch / "feature_config.yaml")
    _atomic_replace(cand_model, cur_model)
    _atomic_replace(cand_cfg, cur_cfg)


def _write_manifest(models_dir: Path, result: dict) -> None:
    (models_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run(
    daily_dir: str | Path,
    eval_normal: str | Path,
    eval_anomalies: str | Path,
    config_path: str | Path,
    models_dir: str | Path,
    window_days: int = 30,
    end_date: date | None = None,
    tolerance: float = 0.01,
    min_rows: int = 200,
) -> dict:
    """파이프라인 1회 실행. 결과 dict 반환(`exit` 키에 종료 코드 후보 포함)."""
    models_dir = Path(models_dir)
    end_date = end_date or date.today()

    df, used = collect_recent(daily_dir, window_days, end_date)
    used_files = [str(p) for p in used]
    if len(df) < min_rows:
        result = {
            "decision": "error",
            "reason": f"학습 데이터 부족 rows={len(df)} (<{min_rows})",
            "window_days": window_days,
            "end_date": end_date.isoformat(),
            "used_files": used_files,
            "exit": 1,
        }
        models_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest(models_dir, result)
        return result

    # 2) 후보 학습
    cand_dir = models_dir / "candidates" / end_date.isoformat()
    cand_dir.mkdir(parents=True, exist_ok=True)
    acc_path = cand_dir / "_accumulated.csv"
    df.to_csv(acc_path, index=False)
    cand_bundle = train(data_path=acc_path, config_path=Path(config_path), out_dir=cand_dir)
    config = FeatureConfig.load(config_path)

    # 3) 검증: 후보 vs 현행
    cand_metrics = evaluate(cand_bundle, config, eval_normal, eval_anomalies)
    cur_model = models_dir / "model.pkl"
    cur_metrics = None
    if cur_model.exists():
        cur_bundle = ModelBundle.load(cur_model)
        cur_cfg = FeatureConfig.load(models_dir / "feature_config.yaml")
        cur_metrics = evaluate(cur_bundle, cur_cfg, eval_normal, eval_anomalies)

    passed, reasons = gate(cand_metrics, cur_metrics, tolerance)

    result = {
        "end_date": end_date.isoformat(),
        "window_days": window_days,
        "rows": int(len(df)),
        "used_files": used_files,
        "candidate_metrics": cand_metrics,
        "current_metrics": cur_metrics,
        "tolerance": tolerance,
        "reasons": reasons,
        "candidate_trained_at": cand_bundle.trained_at,
    }

    # 4/5) 승격 or 거부
    if passed:
        _promote(cand_dir / "model.pkl", cand_dir / "feature_config.yaml", models_dir, end_date)
        result["decision"] = "promoted"
        result["exit"] = 0
    else:
        result["decision"] = "rejected"
        result["exit"] = 2

    _write_manifest(models_dir, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="데일리 누적 재학습 파이프라인")
    p.add_argument("--daily-dir", type=Path, default=Path("data/daily"))
    p.add_argument("--eval-normal", type=Path, default=Path("data/eval/normal.csv"))
    p.add_argument("--eval-anomalies", type=Path, default=Path("data/eval/anomalies.csv"))
    p.add_argument("--config", type=Path, default=Path("config/feature_config.yaml"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--date", type=str, default=None, help="기준일 YYYY-MM-DD (기본: 오늘)")
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--min-rows", type=int, default=200)
    args = p.parse_args()

    end_date = date.fromisoformat(args.date) if args.date else date.today()
    result = run(
        daily_dir=args.daily_dir,
        eval_normal=args.eval_normal,
        eval_anomalies=args.eval_anomalies,
        config_path=args.config,
        models_dir=args.models_dir,
        window_days=args.window_days,
        end_date=end_date,
        tolerance=args.tolerance,
        min_rows=args.min_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["decision"] == "promoted":
        print("\n[승격됨] 추론 백엔드를 재시작해 새 모델을 로드하세요 "
              "(예: docker restart inference).", file=sys.stderr)
    elif result["decision"] == "rejected":
        print("\n[거부됨] 현행 모델 유지. 후보/사유는 manifest.json 참고.", file=sys.stderr)
    sys.exit(result["exit"])


if __name__ == "__main__":
    main()
