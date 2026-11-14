FROM python:3.12.8-slim@sha256:2199a62885a12290dc9c5be3ca0681d367576ab7bf037da120e564723292a2f0
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MWS_MODE=mock
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock && useradd --uid 10001 --create-home app
COPY src ./src
COPY frontend ./frontend
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
