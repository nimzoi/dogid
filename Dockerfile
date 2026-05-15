# Multi-stage build dla mniejszego obrazu finalnego.
# Stage "builder" instaluje cieżkie zależności (~2 GB),
# stage "runtime" kopiuje już tylko gotowe biblioteki Pythona.

FROM python:3.11-slim AS builder

WORKDIR /build

# Niezbędne biblioteki systemowe dla niektórych pakietów Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# PyTorch z indexu CPU-only (oszczędza ~1 GB wersji CUDA)
RUN pip install --user --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision && \
    pip install --user --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

# Kod aplikacji (data/raw, data/processed wykluczone w .dockerignore)
COPY data/ ./data/
COPY model/ ./model/
COPY app/ ./app/
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8501 8000

# Domyślnie uruchamia Streamlit. docker-compose nadpisuje to dla serwisu API.
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
