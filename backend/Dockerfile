# Backend 추론 서비스 이미지
# 빌드 컨텍스트는 프로젝트 루트(kpi-ai-project)에서:
#   docker build -f backend/Dockerfile -t anomaly-backend .
FROM python:3.11-slim

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 공유 코어 + 백엔드 + 학습 산출물 복사
COPY core /app/core
COPY backend /app/backend
COPY models /app/models

ENV MODEL_PATH=/app/models/model.pkl \
    FEATURE_CONFIG_PATH=/app/models/feature_config.yaml \
    MQTT_ENABLED=true \
    MQTT_HOST=mqtt \
    MQTT_PORT=1883

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
