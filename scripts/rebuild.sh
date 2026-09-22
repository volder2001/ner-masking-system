#!/bin/bash
set -e
echo "🔄 Пересборка и перезапуск сервисов..."
docker compose down
echo "🔨 Сборка образов..."
docker compose build --no-cache
docker compose up -d
echo "✅ Пересборка завершена!"
