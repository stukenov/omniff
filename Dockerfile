FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app
COPY python/ python/
COPY README.md LICENSE ./

RUN pip install --no-cache-dir python/.[all]

ENV PYTHONPATH=/app/python
EXPOSE 8000 7860

CMD ["uvicorn", "omniff.api:app", "--host", "0.0.0.0", "--port", "8000"]
