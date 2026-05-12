FROM python:3.11-slim

WORKDIR /app

# Instalo só as dependências da API (build mais rápido e imagem menor).
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

# O Railway injeta a porta via variável PORT; localmente cai pra 8000.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
