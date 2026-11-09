FROM --platform=linux/amd64 python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /processor

# Pinned Python dependencies (manylinux wheels only, no compilers needed).
COPY requirements.txt /processor/requirements.txt
RUN pip install --only-binary=:all: -r /processor/requirements.txt

COPY processor/ /processor/processor/
COPY VERSION LICENSE THIRD_PARTY.md /processor/

# The run-time user (uid 1000) with a passwd entry; everything it writes goes to /tmp or OUTPUT_DIR.
RUN useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin processor

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba

# Build matplotlib's font cache as the run-time user would see it, and check the stack imports.
RUN python -c "import matplotlib.pyplot, rastermap, sklearn.decomposition, scipy.stats, h5py, pandas" \
    && mkdir -p /tmp/numba && chmod -R a+rwX /tmp/matplotlib /tmp/numba

USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
