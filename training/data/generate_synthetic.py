"""합성 CAN 데이터 생성기.

실제 CAN 정의서가 확정되기 전, 학습/테스트용 더미 데이터를 만든다.
정상 운행 패턴을 기본으로 하고, 옵션으로 이상 구간(과열/과전류/압력저하/
에러폭증 등)을 주입해 end-to-end 검증에 사용한다.

CLI:
  python -m src.data.generate_synthetic --minutes 60 --out data/normal.csv
  python -m src.data.generate_synthetic --minutes 10 --with-anomalies \
      --out data/test_with_anomalies.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 정상 운행 시 각 신호의 평균/표준편차 (더미 기준값)
NORMAL = {
    "rpm": (1500.0, 80.0),
    "speed": (8.0, 1.0),
    "battery_voltage": (48.0, 0.4),
    "motor_current": (30.0, 4.0),
    "motor_temperature": (55.0, 3.0),
    "hydraulic_pressure": (120.0, 5.0),
}


def _base_signals(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """정상 패턴 + 완만한 운행 변동(사인파)으로 기본 신호 생성."""
    t = np.arange(n)
    duty = 0.5 + 0.5 * np.sin(2 * np.pi * t / max(n, 1) * 3)  # 부하 사이클
    data: dict[str, np.ndarray] = {}
    for name, (mean, std) in NORMAL.items():
        noise = rng.normal(0.0, std, n)
        if name in ("rpm", "speed", "motor_current"):
            # 부하에 따라 변동하는 신호
            data[name] = mean * (0.85 + 0.3 * duty) + noise
        elif name == "motor_temperature":
            # 부하 누적에 약하게 비례
            data[name] = mean + 8.0 * duty + noise
        else:
            data[name] = mean + noise
    # error_count: 정상 시 대부분 0, 드물게 1
    data["error_count"] = (rng.random(n) < 0.002).astype(float)
    return data


def _inject_anomalies(
    data: dict[str, np.ndarray], n: int, rng: np.random.Generator
) -> None:
    """여러 종류의 이상 구간을 in-place 로 주입."""
    def seg(frac_start: float, length: int) -> slice:
        start = int(n * frac_start)
        return slice(start, min(start + length, n))

    dur = max(n // 25, 5)

    # 1) 모터 과열
    s = seg(0.20, dur)
    data["motor_temperature"][s] += np.linspace(20, 45, s.stop - s.start)

    # 2) 과전류 + rpm 급변
    s = seg(0.45, dur)
    data["motor_current"][s] += 40
    data["rpm"][s] += rng.normal(0, 400, s.stop - s.start)

    # 3) 유압 급락
    s = seg(0.65, dur)
    data["hydraulic_pressure"][s] -= 60

    # 4) 에러 폭증 + 배터리 전압 강하
    s = seg(0.85, dur)
    data["error_count"][s] = rng.integers(1, 5, s.stop - s.start).astype(float)
    data["battery_voltage"][s] -= 6


def generate(
    minutes: float = 60.0,
    sample_rate_hz: int = 10,
    with_anomalies: bool = False,
    seed: int = 42,
    start: str = "2026-01-01T00:00:00",
) -> pd.DataFrame:
    """합성 CAN 데이터프레임 생성."""
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * sample_rate_hz)
    if n <= 0:
        raise ValueError("minutes/sample_rate_hz 가 너무 작습니다.")

    signals = _base_signals(n, rng)
    if with_anomalies:
        _inject_anomalies(signals, n, rng)

    period_ms = int(1000 / sample_rate_hz)
    timestamps = pd.date_range(start=start, periods=n, freq=f"{period_ms}ms")

    df = pd.DataFrame({"timestamp": timestamps, **signals})
    # 물리적으로 음수가 불가능한 신호 클리핑
    for col in ("rpm", "speed", "motor_current", "hydraulic_pressure", "error_count"):
        df[col] = df[col].clip(lower=0)
    return df


def _save(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="합성 CAN 데이터 생성")
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--sample-rate-hz", type=int, default=10)
    p.add_argument("--with-anomalies", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    df = generate(
        minutes=args.minutes,
        sample_rate_hz=args.sample_rate_hz,
        with_anomalies=args.with_anomalies,
        seed=args.seed,
    )
    _save(df, args.out)
    print(f"생성 완료: {args.out}  (rows={len(df)}, anomalies={args.with_anomalies})")


if __name__ == "__main__":
    main()
