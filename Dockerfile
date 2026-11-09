FROM --platform=linux/amd64 python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/ophys-activity

# Pinned Python dependencies (manylinux wheels only, no compilers needed).
COPY requirements.txt /opt/ophys-activity/requirements.txt
RUN pip install --only-binary=:all: -r /opt/ophys-activity/requirements.txt

COPY processor/ /opt/ophys-activity/processor/
COPY VERSION LICENSE THIRD_PARTY.md /opt/ophys-activity/

# uid 1000 with a passwd entry for ECS; on Lambda the app runs as an arbitrary user, so USER is
# set for getpass and nothing depends on owning files in the image.
RUN useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && python -c "import rastermap, sklearn.decomposition, scipy.stats, h5py, pandas, matplotlib"

# Caches that must be written (matplotlib, numba) are put under $TMPDIR at startup by
# processor/main.py; nothing is baked into /tmp or HOME, which are replaced at run time.
ENV PYTHONPATH="/opt/ophys-activity" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    USER=app

USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
