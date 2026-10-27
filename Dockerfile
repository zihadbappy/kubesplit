# One image for every role; KS_ROLE or the first argument picks server / client / prepare-data.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only torch from PyTorch's own index: the default PyPI wheel drags in
# several GB of CUDA libraries. Its own layer, so code changes don't reinstall it.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0 torchvision==0.26.0

COPY pyproject.toml ./
COPY kubesplit ./kubesplit
RUN pip install .

RUN useradd --uid 10001 --create-home ks \
    && mkdir -p /data /results \
    && chown ks /data /results
USER 10001

ENV KS_DATA_ROOT=/data \
    KS_RESULTS_DIR=/results
# 50051 = gRPC (server only), 8000 = Prometheus /metrics (every role)
EXPOSE 50051 8000
ENTRYPOINT ["python", "-m", "kubesplit"]
