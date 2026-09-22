# ВАЖНО: Используем Python 3.10, так как Natasha и её зависимости стабильны именно на нем
FROM python:3.10-slim

WORKDIR /app

# 1. Принудительно ставим версию setuptools, где pkg_resources ЕЩЁ существует
RUN pip install --no-cache-dir --upgrade pip "setuptools<70.0.0"

# 2. Системные зависимости для компиляции
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем код
COPY . .

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]