#!/bin/bash
echo "🧹 Очистка временных данных..."
docker compose down
read -p "⚠️  Удалить все данные (volumes)? (y/N): " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    docker compose down -v
    echo "✅ Volumes удалены"
else
    echo "ℹ️  Volumes сохранены"
fi
docker image prune -f
echo "✅ Очистка завершена!"
