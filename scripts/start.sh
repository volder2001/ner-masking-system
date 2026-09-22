#!/bin/bash
set -e
echo "🚀 Запуск NER Masking System..."

if docker compose ps 2>/dev/null | grep -q "Up"; then
    echo "⚠️  Сервисы уже запущены."
    docker compose ps
    exit 0
fi

docker compose up -d

echo ""
echo "✅ Все сервисы запущены!"
echo ""
echo "📊 Доступные сервисы:"
echo "   🔹 API Gateway:              http://localhost:8000"
echo "   🔹 Document Service:         http://localhost:8001"
echo "   🔹 NER Service:              http://localhost:8002"
echo "   🔹 Masking Service:          http://localhost:8003"
echo "   🔹 Entity Dictionary:        http://localhost:8004"
echo "   🔹 MinIO Console:            http://localhost:9001"
echo "      (логин: minio_user / пароль: minio_password)"
