# CellScanner Docker image (GUI and CLI)
#
# Build (tag with the version in cellscanner/scripts/__init__.py):
#   VERSION=$(python -c "exec(open('cellscanner/scripts/__init__.py').read()); print(__version__)")
#   docker build --build-arg CELLSCANNER_VERSION=$VERSION -t cellscanner:$VERSION -t cellscanner:latest .
#
# Run the GUI (Linux; allow local X connections first with `xhost +local:`):
#   docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ./Testfiles:/media cellscanner
#
# Run the CLI (no display needed; paths in the config refer to the container, e.g. /media/...):
#   docker run --rm -v ./Testfiles:/media cellscanner python CellscannerCLI.py -c /media/config.yml
#
# Input files and outputs live in the directory mounted at /media: when started from /app,
# CellScanner writes its findings there.

FROM python:3.12-slim-bookworm

ARG CELLSCANNER_VERSION=unknown
LABEL org.opencontainers.image.title="CellScanner" \
      org.opencontainers.image.version="${CELLSCANNER_VERSION}" \
      org.opencontainers.image.source="https://github.com/msysbio/CellScanner"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libraries needed by the Qt (PyQt5) GUI to open windows through X11
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libdbus-1-3 \
    libfontconfig1 \
    libx11-xcb1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    && rm -rf /var/lib/apt/lists/*

# CellScanner checks for /app to know it runs in the container
WORKDIR /app

# Install Python dependencies first, so code changes do not invalidate this layer
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY cellscanner ./

# Writable cache locations, so the container also runs as a non-root user (e.g. `--user $(id -u):$(id -g)`, Apptainer):
# UMAP's numba would otherwise try to cache compiled code inside the read-only site-packages
ENV NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

# Default: the GUI. Override the command to run the CLI (see above).
CMD ["python", "Cellscanner.py"]
