# Dockerfile - containerises the Image Classification serving app
# Author: Miraat Gupta | PRN 23070126073 | AIML - A3
#
# The image is built by the GitHub Actions CI/CD workflow (ci-cd.yaml)
# after "dvc pull" has fetched the trained artefacts from the AWS S3 DVC
# remote, so the model files below exist in the build context even though
# they are not stored in git.

# Same Python minor version as the project venv that trained the model
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so this layer is cached between builds
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Serving code
COPY app/ app/

# Pipeline artefacts (DVC-tracked, pulled from S3 before the build)
COPY models/model.joblib models/model.joblib
COPY data/features/selector.joblib data/features/selector.joblib
COPY data/processed/X_test.npy data/processed/X_test.npy
COPY data/processed/y_test.npy data/processed/y_test.npy
COPY reports/metrics.json reports/metrics.json

# Build label shown on the web page and /health (set by the workflow)
ARG APP_VERSION=local
ENV APP_VERSION=${APP_VERSION}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
