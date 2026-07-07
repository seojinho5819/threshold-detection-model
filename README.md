# AI 이상탐지 시스템

중장비/방제로봇의 CAN 데이터 기반 **실시간 이상탐지 + Health Score** 시스템.
**학습 파이프라인**(feature engineering → Isolation Forest 학습 →
`model.pkl` / `feature_config.yaml`)과 **Backend**(FastAPI + MQTT 실시간 추론),
그리고 **합성 데이터 생성기**로 구성된다.

> 예지정비/고장시점예측이 아닌, 실시간 이상탐지와 Health Score 산출이 목표.

## 디렉토리 구조 (monorepo)

학습 서버와 Backend 는 **추론 코드를 공유**하므로 별도 프로젝트로 쪼개지 않고
한 레포 안에서 `core`(공유) / `training` / `backend` 패키지로 분리한다.
배포 시 Backend 만 별도 Docker 이미지로 묶는다.

```
kpi-ai-project/
├── config/feature_config.yaml   # Feature 정의 (이 파일만 고치면 구성 변경)
├── core/                        # ★ 공유 코어 (training·backend 공용)
│   ├── config.py                #   feature_config 로딩 + 검증
│   ├── features/feature_engineering.py  # 최근 N초 통계 Feature (학습·추론 공용)
│   ├── models/{base,isolation_forest,registry}.py  # 모델 추상화 + IF 구현
│   ├── health_score.py          #   anomaly score → 0~100 + 상태 판정
│   └── bundle.py                #   model.pkl 직렬화 (모델+보정+메타)
├── training/                    # 학습 파이프라인
│   ├── data/generate_synthetic.py  # 합성 CAN 데이터 생성기
│   ├── train.py                 #   학습 엔트리포인트
│   └── predict.py               #   오프라인 추론 검증
├── backend/                     # 실시간 추론 서비스
│   ├── app.py                   #   FastAPI (lifespan: 모델 로드 + MQTT 기동)
│   ├── inference.py             #   InferenceService (MQTT·HTTP 공통 진입점)
│   ├── buffer.py                #   로봇별 최근 N초 슬라이딩 버퍼
│   ├── mqtt_client.py           #   MQTT 수신 → ingest
│   ├── simulate.py              #   CSV → MQTT 시뮬레이터 (실연동 테스트)
│   ├── settings.py / requirements.txt / Dockerfile
├── tests/{test_pipeline,test_backend}.py
└── requirements.txt
```

## 설치

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 사용법 (end-to-end)

```powershell
$PY = ".\.venv\Scripts\python.exe"

# 1) 합성 데이터 생성 (실제 CAN 정의서 확정 전 더미)
& $PY -m training.data.generate_synthetic --minutes 60 --out data/normal.csv
& $PY -m training.data.generate_synthetic --minutes 10 --with-anomalies --seed 7 `
      --out data/test_with_anomalies.csv

# 2) 학습 → models/model.pkl, models/feature_config.yaml 생성
& $PY -m training.train --data data/normal.csv --config config/feature_config.yaml `
      --out-dir models

# 3) 추론 검증 (정상 파일은 대부분 normal, 이상 파일은 critical 구간 발생)
& $PY -m training.predict --data data/test_with_anomalies.csv --robot-id robot-001

# 4) 테스트
& $PY -m pytest -q
```

## Backend (실시간 추론)

```powershell
# MQTT 브로커 없이 API 만 기동 (개발/테스트)
$env:MQTT_ENABLED = "false"
& $PY -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

# 브로커가 있으면 (기본 MQTT_ENABLED=true), Agent 가 robots/<id>/can 으로 publish
$env:MQTT_HOST = "localhost"
& $PY -m uvicorn backend.app:app --port 8000
# 다른 터미널에서 CSV 를 MQTT 로 흘려보내 실연동 테스트
& $PY -m backend.simulate --data data/test_with_anomalies.csv --robot-id robot-001 --speed 50
```

엔드포인트:

| 메서드·경로 | 설명 |
|---|---|
| `GET /health` | 서비스/모델/MQTT 상태 |
| `GET /robots` | 전체 로봇 최신 Health (Frontend 장비 목록용) |
| `GET /robots/{id}` | 특정 로봇 최신 결과 |
| `POST /ingest/{id}` | CAN 샘플 1건 HTTP 주입 (MQTT 대체/테스트) |

MQTT 수신과 HTTP `/ingest` 는 **동일한 `InferenceService`** 를 거친다.
MQTT 브로커가 없으면 경고만 남기고 API 는 정상 기동한다(graceful degradation).

**추론 결과 발행**: MQTT 수신 경로에서는 매 추론 결과를 `robots/<id>/health`
토픽으로 다시 publish 한다(QoS1, retain). 웹 백엔드가 이를 구독해 이상
이벤트 저장/알림을 담당하며, 추론 백엔드는 발행까지만 하고 저장·판단은 하지
않는다(무상태). 토픽/QoS/retain 은 `MQTT_HEALTH_TOPIC` /`MQTT_HEALTH_QOS` /
`MQTT_HEALTH_RETAIN` 환경변수로 조정한다.

### Docker

```powershell
# 빌드 컨텍스트는 프로젝트 루트
docker build -f backend/Dockerfile -t anomaly-backend .
docker run -p 8000:8000 -e MQTT_HOST=<broker> anomaly-backend
```

### ⚠️ 중요 제약: 수신 주기 = 학습 주기

Feature 가 "최근 N초 윈도우의 통계(std 등)"이므로 **윈도우 내 샘플 밀도**가
학습/추론 간에 일치해야 한다. 학습 데이터가 10Hz 였다면 Agent 도 ~10Hz 로
보내야 한다. 주기가 크게 다르면 std 통계가 흔들려 정상 데이터도 이상으로
오탐될 수 있다. (검증: 10Hz→2Hz 다운샘플 시 normal 비율 99%→62% 하락)

### 워밍업 (콜드스타트 오탐 방지)

로봇 접속 직후 버퍼가 거의 비어 있으면(std=0 등) 학습 분포와 어긋나 오탐이
난다. 버퍼가 윈도우의 절반(`WARMUP_MIN_SPAN_SECONDS`, 기본 = window/2)만큼
시간 구간을 채우기 전에는 점수 대신 `status: "warmup"`, `health_score: null`
을 반환한다.

## 데일리 누적 재학습 (`training/daily_retrain.py`)

에이전트 개발자가 매일 정제해 떨구는 정상 데이터 파일을 누적해, **정해진 시각에
자동 재학습 → 검증 → 승격**하는 오케스트레이터. Isolation Forest 는 incremental
학습이 불가하므로 "누적 학습"=매번 최근 N일 데이터로 from-scratch 재학습이다.
재학습이 정확도를 올리는 원리는 **정상 envelope 커버리지 확대(오탐 감소)** 와
**드리프트 적응**이다.

```
data/daily/YYYY-MM-DD.csv  (에이전트가 매일 떨굼, config 와 동일 스키마)
data/eval/{normal,anomalies}.csv  (고정 평가셋 — 절대 안 바뀌는 비교 기준)
```

```powershell
& $PY -m training.daily_retrain --daily-dir data/daily `
      --eval-normal data/eval/normal.csv --eval-anomalies data/eval/anomalies.csv `
      --window-days 30 --models-dir models
```

흐름: **최근 N일 concat → 후보 학습(candidates/<date>) → 고정 eval셋으로 후보 vs
현행 비교 → 악화 없을 때만 원자적 승격(+archive 백업 + manifest.json)**.

| 검증 지표 | 의미 | 통과 조건 |
|---|---|---|
| `normal_ratio` | 정상셋의 normal 비율 (오탐 적을수록 ↑) | 현행 − `tolerance` 이상 |
| `detect_ratio` | 이상셋의 비-normal 비율 (미탐 적을수록 ↑) | 현행 − `tolerance` 이상 |

현행 모델이 없으면 부트스트랩으로 무조건 승격. 라벨은 학습이 아니라 **eval셋
(검증)** 에만 쓴다 — 비지도 모델은 절대 정확도를 못 재므로 고정셋 상대 비교로 판단.

종료 코드: `0` 승격(스케줄러가 이어서 추론 백엔드 재시작), `2` 거부(현행 유지),
`1` 데이터 부족. 스케줄링은 OS(작업 스케줄러/cron)로 매일 특정 시각 실행하고,
승격(exit 0) 시에만 추론 백엔드를 재시작하도록 연결한다.

> 데이터 스키마/주기/파일 규약은 `docs/data-contract.md` (에이전트 개발자 전달용) 참조.

## 산출물

- **`model.pkl`** — 학습된 detector(스케일러 포함) + Feature 순서 + window 크기 +
  Health 보정 파라미터 + 메타. Backend 가 `ModelBundle.load()` 한 번으로 추론 가능.
- **`feature_config.yaml`** — 학습에 사용된 Feature 정의 (Backend 배포용 복사본).
- **`manifest.json`** — 마지막 재학습의 데이터 범위·메트릭·승격 여부 기록.

## Feature Engineering

원본 CAN 신호를 그대로 쓰지 않고 **최근 N초 구간의 통계 Feature**를 만든다.
정의는 `config/feature_config.yaml` 의 `features` 목록으로 관리하며,
지원 집계는 `mean / std / max / min / sum / last`.

```yaml
window_seconds: 10
features:
  - { name: rpm_mean,        source: rpm,               agg: mean }
  - { name: temperature_max, source: motor_temperature, agg: max }
```

학습은 파일 전체에 슬라이딩 윈도우(`build_features`)를, 추론은 버퍼 1개에
`compute_features_from_buffer` 를 적용한다 — **동일한 정의를 공유**한다.

## Health Score 보정 (핵심 설계)

anomaly score(높을수록 정상)를 0~100으로 변환할 때, 단순 min-max 선형 매핑은
정상 데이터 **내부의 미세한 점수 편차**까지 health 하락으로 잘못 반영한다.

대신 **모델의 판정 경계(IsolationForest `decision_function = 0`)**를 기준으로 앵커링한다.

| 조건 | Health |
|------|--------|
| `score ≥ boundary + pos_scale` | 100 |
| `score == boundary` | 90 (normal/warning 경계) |
| `score ≤ boundary − neg_scale` | 0 |

- `pos_scale` = 정상 점수 상위 percentile − boundary
- `neg_scale` = 정상 산포만큼 경계 아래로 벗어나면 health 0 (보수적 기준)

결과적으로 정상 데이터는 대부분 90~100(normal)에 모이고, 모델이 이상으로 본
구간만 warning/critical 로 떨어진다. (검증: 정상 파일 99.0% normal,
`contamination=0.01` 과 일치)

상태 구간: `90~100 normal`, `70~89 warning`, `0~69 critical`.

출력 규격:

```json
{ "robot_id": "robot-001", "health_score": 91, "anomaly_score": 0.12, "status": "normal" }
```

## 향후 확장 (AutoEncoder 등)

`AnomalyDetector` 인터페이스(`fit` / `score`, "높을수록 정상")만 구현하고
`core/models/registry.py` 에 등록하면 교체 가능. 학습/추론/Health 로직과
Backend 는 그대로 재사용된다 (Backend 는 `ModelBundle` +
`compute_features_from_buffer` + `build_result` 만 사용).

## 남은 작업

- **Frontend**: `GET /robots` 를 폴링해 장비 목록/상태(Normal·Warning·Critical) 표시.
- **CAN 정의서 반영**: 확정되면 `config/feature_config.yaml` 의 신호/Feature 교체 후 재학습.
- **하이퍼파라미터 튜닝**: 실제 데이터로 `contamination`, `window_seconds`, 보정 percentile 재조정.
- **운영**: 브로커(mosquitto) docker-compose 구성, 모델 버전 관리/재배포 절차.
