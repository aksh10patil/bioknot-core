FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 2 --timeout 120 app:app"]
