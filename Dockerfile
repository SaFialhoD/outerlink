# OuterLink Development Container
# Rust + CUDA stub headers + build tools
FROM rust:1.85-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libclang-dev \
    clang \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust components
RUN rustup component add clippy rustfmt
RUN cargo install cargo-watch

# Create CUDA stub headers (for compilation without real CUDA)
RUN mkdir -p /usr/local/cuda/include
COPY cuda-stubs/ /usr/local/cuda/include/

ENV CUDA_PATH=/usr/local/cuda
ENV PATH="${CUDA_PATH}/bin:${PATH}"

WORKDIR /workspace

# Cache dependencies by copying Cargo files first
COPY Cargo.toml Cargo.lock* ./
COPY crates/ ./crates/

# Build dependencies (cached layer)
RUN cargo build 2>/dev/null || true

# Copy full source
COPY . .

CMD ["cargo", "build", "--all"]
