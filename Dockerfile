FROM ubuntu:20.04

# Set environment variables to make the build non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Update and install prerequisites for adding the PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    lsb-release \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip using the official get-pip.py script
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN curl -O https://bootstrap.pypa.io/get-pip.py && python get-pip.py 

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update && \
    # apt-get install -y libx11-dev libgl1-mesa-dev && \
    apt-get install -y libx11-dev libgl1-mesa-dev libxcomposite-dev libxrandr-dev libxss-dev libxcursor-dev

RUN apt-get install -y libx11-xcb1 libxcb-xinerama0 libxkbcommon0 libglib2.0-0


RUN apt-get update && apt-get install -y \
    libx11-xcb1 \
    libxcb-util1 \
    # libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    x11-utils \
    libxkbcommon-x11-0

RUN export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/local/lib/python3.10/dist-packages/PyQt5/Qt5/plugins/platforms
RUN export QT_DEBUG_PLUGINS=1


# Copy Cellscanner files to the working directory
COPY cellscanner ./

# Specify the command to run the application (optional)
CMD ["python", "Cellscanner.py"]



# docker run --rm -it --entrypoint /bin/bash cellscanner
# docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v .:/media cellscanner




# # FROM DEEPSEEK AS ALTERNATIVE -- NOT WORKING MOUHAHA
# FROM ubuntu:20.04

# # Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive \
#     TZ=Etc/UTC \
#     QT_QPA_PLATFORM_PLUGIN_PATH=/usr/local/lib/python3.10/dist-packages/PyQt5/Qt5/plugins/platforms \
#     QT_DEBUG_PLUGINS=1

# # Combine all package installations into a single layer
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     software-properties-common \
#     build-essential \
#     libssl-dev \
#     libffi-dev \
#     curl \
#     lsb-release \
#     python3.10 \
#     python3.10-dev \
#     python3.10-venv \
#     python3.10-distutils \
#     libx11-dev \
#     libgl1-mesa-dev \
#     libxcomposite-dev \
#     libxrandr-dev \
#     libxss-dev \
#     libxcursor-dev \
#     libx11-xcb1 \
#     libxcb-xinerama0 \
#     libxkbcommon0 \
#     libglib2.0-0 \
#     libxcb-util1 \
#     libxcb-icccm4 \
#     libxcb-image0 \
#     libxcb-keysyms1 \
#     libxcb-randr0 \
#     libxcb-render-util0 \
#     libxcb-render0 \
#     libxcb-shape0 \
#     libxcb-shm0 \
#     libxcb-sync1 \
#     libxcb-xfixes0 \
#     libxcb-xkb1 \
#     x11-utils \
#     libxkbcommon-x11-0 && \
#     add-apt-repository -y ppa:deadsnakes/ppa && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# # Install pip and setup Python
# RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
#     curl -sS https://bootstrap.pypa.io/get-pip.py | python -

# WORKDIR /app

# # Install Python dependencies first (better layer caching)
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application files
# COPY cellscanner ./

# CMD ["python", "Cellscanner.py"]
