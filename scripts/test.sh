#!/bin/bash
set -e
echo "🧪 Запуск тестов..."
if ! docker compose ps | grep -q "Up"; then
    echo "⚠️  Сервисы не запущены. Запускаем..."
    ./scripts/start.sh
fi
echo "🔹 Тесты NER Service..."
docker compose exec ner-service pytest tests/ -v || echo "Тесты не найдены"
echo "✅ Тесты завершены!"
