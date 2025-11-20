# ================================
# Stage 1: Builder - 构建环境
# ================================
FROM python:3.12-slim AS builder

# 设置工作目录
WORKDIR /build

# 使用国内镜像源加速（阿里云）- 强制替换配置
RUN echo "===> Configuring Aliyun mirror..." && \
    rm -f /etc/apt/sources.list.d/debian.sources && \
    echo "deb http://mirrors.aliyun.com/debian/ trixie main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ trixie-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security/ trixie-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "===> Mirror configuration:" && \
    cat /etc/apt/sources.list

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv (快速的 Python 包管理器)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    (ln -sf /root/.local/bin/uv /usr/local/bin/uv || \
     ln -sf /root/.cargo/bin/uv /usr/local/bin/uv) && \
    uv --version
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

# 复制依赖配置文件
COPY pyproject.toml uv.lock ./

# 安装依赖到 /build/.venv
RUN /usr/local/bin/uv sync --frozen --no-dev

# 复制源代码
COPY src ./src
COPY config ./config
COPY main.py ./
COPY README.md ./

# ================================
# Stage 2: Runtime - 运行环境
# ================================
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 使用国内镜像源加速（阿里云）- 强制替换配置
RUN echo "===> Configuring Aliyun mirror..." && \
    rm -f /etc/apt/sources.list.d/debian.sources && \
    echo "deb http://mirrors.aliyun.com/debian/ trixie main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ trixie-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security/ trixie-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "===> Mirror configuration:" && \
    cat /etc/apt/sources.list

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app

# 从 builder 复制虚拟环境
COPY --from=builder --chown=trader:trader /build/.venv /app/.venv

# 从 builder 复制应用代码
COPY --from=builder --chown=trader:trader /build/src /app/src
COPY --from=builder --chown=trader:trader /build/config /app/config
COPY --from=builder --chown=trader:trader /build/main.py /app/
COPY --from=builder --chown=trader:trader /build/README.md /app/

# 创建必要的目录
# 注意：LangChain v1.0 使用内存存储，不再需要大量磁盘空间存储 checkpoint
RUN mkdir -p /app/logs /app/data && chown -R trader:trader /app/logs /app/data

# 设置环境变量
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# 切换到非 root 用户
USER trader

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# 暴露端口 (如果需要 API 服务)
# EXPOSE 8000

# 默认启动命令
CMD ["python", "main.py"]
