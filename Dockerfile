FROM python:3.11-slim
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx libgomp1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user \
    && chown user:user /app
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user:user . .
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh
USER user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EXPLAINMAINT_API_URL=http://127.0.0.1:8000 \
    EXPLAINMAINT_API_ROOT_PATH=/api
RUN mkdir -p artifacts data
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD ["python", "deployment/healthcheck.py"]
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "./start.sh"]
