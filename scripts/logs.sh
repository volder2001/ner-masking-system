#!/bin/bash
if [ -z "$1" ]; then
    echo "📋 Использование: ./scripts/logs.sh [сервис]"
    echo ""
    echo "Доступные сервисы:"
    echo "  api-gateway"
    echo "  document-service"
    echo "  ner-service"
    echo "  masking-service"
    echo "  entity-dictionary-service"
    echo "  postgres"
    echo "  redis"
    echo "  minio"
    exit 0
fi
echo "📋 Логи сервиса: $1"
echo "─────────────────────────────────────"
docker compose logs -f "$1"
