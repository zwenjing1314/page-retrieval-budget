FROM huggingface/transformers-pytorch-gpu:latest@sha256:4c7317881a534b22e18add49c925096fa902651fb0571c69f3cad58af3ea2c0f

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl git poppler-utils \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface \
    HF_DATASETS_CACHE=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    STREAMLIT_CONFIG_HOME=/app/.streamlit \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_USER_BASE_PATH=/app/.cache/streamlit

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

RUN mkdir -p /app/.streamlit /app/.cache/streamlit /app/.cache/huggingface /app/results && \
    printf '[server]\nport = 7860\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\nmaxUploadSize = 500\n\n[browser]\ngatherUsageStats = false\n' > /app/.streamlit/config.toml && \
    chmod -R 777 /app/.streamlit /app/.cache /app/results

COPY visual_rag/ ./visual_rag/
COPY benchmarks/ ./benchmarks/
COPY demo/ ./demo/
COPY pyproject.toml README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -e .

EXPOSE 7860
ENTRYPOINT ["streamlit", "run", "demo/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless", "true"]