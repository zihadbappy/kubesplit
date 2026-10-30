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

# tc (iproute2), for the init container that adds network delay (run.netDelayMs).
RUN apt-get update \
    && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*

# CIFAR-10 (178 MB), baked in so pods never download it: the canonical host is
# slow. Taken from the build context, so run `python -m kubesplit prepare-data`
# first. Loading it here checks every file's MD5: a missing or corrupt copy
# fails the build, not a pod.
COPY data/cifar-10-batches-py /data/cifar-10-batches-py
RUN python -c "from torchvision.datasets import CIFAR10; \
[CIFAR10('/data', train=t) for t in (True, False)]"

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
