FROM ubuntu:24.04

# x86_64 / amd64 only — pre-built SDSL static libs are x86_64 object archives.
LABEL org.opencontainers.image.authors="tvosch"
LABEL org.opencontainers.image.description="infini-gram-mini indexing + query container (Ubuntu 24.04, GCC 13, Python 3.12)"

# ---------------------------------------------------------------------------
# Source files (mirroring %files in infini_gram_mini.def)
# ---------------------------------------------------------------------------
COPY third_party         /infini-gram-mini/third_party
COPY infini-gram-mini    /infini-gram-mini/src
COPY pyproject.toml      /infini-gram-mini/pyproject.toml

# ---------------------------------------------------------------------------
# Runtime environment (mirroring %environment)
# ---------------------------------------------------------------------------
ENV PYTHONNOUSERSITE=1 \
    PYTHONPATH=/infini-gram-mini/src \
    PATH=/root/.local/bin:/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin

# ---------------------------------------------------------------------------
# Build steps (mirroring %post)
# ---------------------------------------------------------------------------
RUN set -e && \
    \
    # Guard: x86_64 only \
    ARCH=$(uname -m) && \
    if [ "$ARCH" != "x86_64" ]; then \
        echo "ERROR: This container requires x86_64 (detected: $ARCH). Abort." && \
        exit 1; \
    fi && \
    \
    # System packages \
    DEBIAN_FRONTEND=noninteractive apt-get update -y && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        curl \
        ca-certificates \
        python3.12 \
        python3.12-dev \
        libdivsufsort-dev && \
    rm -rf /var/lib/apt/lists/* && \
    \
    # uv \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:${PATH}" && \
    export UV_LINK_MODE=copy && \
    \
    # Rust toolchain (needed to build rust_indexing) \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y \
            --default-toolchain stable \
            --profile minimal \
            --target x86_64-unknown-linux-gnu \
            --no-modify-path && \
    . /root/.cargo/env && \
    \
    # Python packages \
    uv pip install --system --no-cache --break-system-packages \
        pybind11 \
        pyarrow \
        numpy \
        zstandard \
        flask \
        gradio \
        requests && \
    \
    # Build rust_indexing \
    cd /infini-gram-mini/third_party/suffix_array && \
    cargo build --release && \
    cp target/release/rust_indexing /infini-gram-mini/src/indexing/rust_indexing && \
    \
    # Recompile libsdsl.a (parallel_sdsl) with -fPIC \
    cd /infini-gram-mini/third_party/parallel_sdsl/lib && \
    g++ -std=c++17 -O2 -fPIC -fpermissive \
        -I/infini-gram-mini/third_party/parallel_sdsl/include \
        -c *.cpp && \
    rm -f libsdsl.a && ar rcs libsdsl.a *.o && rm -f *.o && \
    \
    # Recompile libsdsl.a (sdsl) with -fPIC \
    cd /infini-gram-mini/third_party/sdsl/lib && \
    g++ -std=c++17 -O2 -fPIC \
        -I/infini-gram-mini/third_party/sdsl/include \
        -c *.cpp && \
    rm -f libsdsl.a && ar rcs libsdsl.a *.o && rm -f *.o && \
    \
    # Build cpp_indexing \
    cd /infini-gram-mini/src/indexing && \
    g++ -std=c++17 -O3 \
        cpp/indexing.cpp \
        -o cpp/cpp_indexing \
        -I/infini-gram-mini/third_party/parallel_sdsl/include \
        -L/infini-gram-mini/third_party/parallel_sdsl/lib \
        -lsdsl -ldivsufsort -ldivsufsort64 && \
    \
    # Build cpp_engine pybind11 extension \
    cd /infini-gram-mini/src/engine && \
    c++ -std=c++17 -O3 -shared -fPIC \
        $(python3.12 -m pybind11 --includes) \
        src/cpp_engine.cpp \
        -o "src/cpp_engine$(python3.12-config --extension-suffix)" \
        -I/infini-gram-mini/third_party/sdsl/include \
        -L/infini-gram-mini/third_party/sdsl/lib \
        -lsdsl -ldivsufsort -ldivsufsort64 -pthread && \
    \
    # Install the Python package \
    ln -s /infini-gram-mini/src /infini-gram-mini/infini-gram-mini && \
    cd /infini-gram-mini && \
    uv pip install --system --no-cache --no-deps --break-system-packages -e . && \
    \
    # Python convenience symlinks \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    \
    # Clean up build artefacts \
    rm -rf /infini-gram-mini/third_party/suffix_array/target \
            /root/.cargo /root/.rustup

WORKDIR /infini-gram-mini/src/indexing

# Default: run the indexing pipeline (pass args via docker run / compose command:)
#   docker run infini-gram-mini --data_dir /data --save_dir /index
ENTRYPOINT ["python3", "jobs/index_parquet.py"]
