#!/bin/bash
set -e
echo "🔧 Первоначальная настройка проекта..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установите Docker Desktop."
    exit 1
fi

mkdir -p data/postgres data/redis data/minio logs
chmod +x scripts/*.sh

echo "✅ Настройка завершена!"
echo "🚀 Теперь запустите: ./scripts/start.sh"
