#!/usr/bin/env bash
# curl 로 백엔드 실시간 추론을 확인하는 간이 데모.
#   1) /health 로 모델 상태 확인
#   2) 실제 정상 데이터(normal.csv) 실데이터 행을 10Hz 로 재생 -> 정상 추론
#   3) /robots/<id> 로 결과 조회 (status: normal 기대)
#   4) 이상 데이터(test_with_anomalies.csv)의 과열 구간 행을 재생 -> health 하락
#   5) /robots 로 전체 목록 조회
#
# 손으로 만든 가짜 값은 학습 분포(구조적 신호 + 변동)를 못 맞춰 오탐된다.
# 그래서 실제 CSV 행을 그대로 흘려보낸다 (분포 100% 일치 보장).
#
# 사용:  bash scripts/curl_demo.sh
# (서버: MQTT_ENABLED=false uvicorn backend.app:app --port 8000 가 떠 있어야 함)
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8000}"
ROBOT="${ROBOT:-robot-001}"
NORMAL_CSV="${NORMAL_CSV:-data/normal.csv}"
ANOM_CSV="${ANOM_CSV:-data/test_with_anomalies.csv}"

# CSV(헤더: timestamp,rpm,speed,battery_voltage,motor_current,
#          motor_temperature,hydraulic_pressure,error_count) 의
# [start, start+count) 데이터 행을 JSON 한 줄씩 출력.
csv_to_json_rows() {
  local file="$1" start="$2" count="$3"
  awk -F, -v s="$start" -v c="$count" '
    NR==1 {next}                       # 헤더 스킵
    (NR-1) > s && (NR-1) <= s+c {
      printf "{\"timestamp\":\"%s\",\"rpm\":%s,\"speed\":%s,\"battery_voltage\":%s,\"motor_current\":%s,\"motor_temperature\":%s,\"hydraulic_pressure\":%s,\"error_count\":%s}\n", \
             $1,$2,$3,$4,$5,$6,$7,$8
    }' "$file"
}

# JSON 행들을 /ingest 로 POST. 재생 중 health_score 가 가장 낮았던(=가장
# 이상에 가까운) 응답을 출력한다. 워밍업(health=null)은 건너뛴다.
replay() {
  local file="$1" start="$2" count="$3"
  local worst="" worst_h=999 resp h
  while IFS= read -r body; do
    resp=$(curl -s -X POST "$BASE/ingest/$ROBOT" -H "Content-Type: application/json" -d "$body")
    h=$(echo "$resp" | sed -n 's/.*"health_score":\([0-9]*\).*/\1/p')
    if [ -n "$h" ] && [ "$h" -lt "$worst_h" ]; then
      worst_h="$h"; worst="$resp"
    fi
  done < <(csv_to_json_rows "$file" "$start" "$count")
  echo "$worst"
}

echo "=== 1) GET /health ==="
curl -s "$BASE/health"; echo

echo
echo "=== 2) 정상 데이터 120행(=12s @10Hz) 재생 ==="
echo "    실데이터를 그대로 흘려보내 학습 분포와 일치시킨다."
last=$(replay "$NORMAL_CSV" 0 120)
echo "  재생 중 최저 health 응답: $last"

echo
echo "=== 3) GET /robots/$ROBOT (정상 추론 결과 -> status: normal 기대) ==="
curl -s "$BASE/robots/$ROBOT"; echo

echo
echo "=== 4) 이상 데이터 과전류구간(행 2680~2940) 재생 -> health 급락(critical) ==="
last=$(replay "$ANOM_CSV" 2680 260)
echo "  재생 중 최저 health 응답: $last"

echo
echo "=== 5) GET /robots (전체 로봇 최신 목록) ==="
curl -s "$BASE/robots"; echo
