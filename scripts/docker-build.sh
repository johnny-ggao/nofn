#!/bin/bash

################################################################################
# Docker 镜像构建脚本
# 用途：构建 Docker 镜像并准备推送到 AWS ECR
################################################################################

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed!${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# 构建参数
BUILD_ARGS=""
NO_CACHE=""
PLATFORM="linux/amd64"  # 默认为 amd64 架构（Ubuntu 服务器）

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --arm64)
            PLATFORM="linux/arm64"
            shift
            ;;
        --amd64)
            PLATFORM="linux/amd64"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-cache] [--platform PLATFORM] [--amd64] [--arm64]"
            exit 1
            ;;
    esac
done

echo "================================"
echo "🐳 Building Docker Image"
echo "================================"
echo ""

# 显示构建信息
echo -e "${BLUE}📦 Build configuration:${NC}"
echo "  - Base image: python:3.12-slim"
echo "  - Target platform: ${PLATFORM}"
echo "  - Package manager: uv"
if [ -n "$NO_CACHE" ]; then
    echo "  - Cache: disabled"
else
    echo "  - Cache: enabled"
fi

echo ""
echo -e "${BLUE}🔨 Starting build...${NC}"

# 记录开始时间
START_TIME=$(date +%s)

# 构建镜像（指定平台）
export DOCKER_DEFAULT_PLATFORM=$PLATFORM
docker compose -f docker/docker-compose.yml build --build-arg BUILDPLATFORM=$PLATFORM $NO_CACHE $BUILD_ARGS

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}✅ Build completed in ${DURATION}s${NC}"

# 显示镜像信息
IMAGE_NAME="nofn-nofn-agent"
IMAGE_SIZE=$(docker images --format "{{.Size}}" $IMAGE_NAME 2>/dev/null | head -1)

echo ""
echo -e "${BLUE}📊 Image:${NC}"
echo "  Name: ${IMAGE_NAME}"
echo "  Platform: ${PLATFORM}"
if [ -n "$IMAGE_SIZE" ]; then
    echo "  Size: ${IMAGE_SIZE}"
fi

echo ""
echo -e "${GREEN}✅ Build complete!${NC}"
echo -e "${BLUE}Next: make push  # 推送到 AWS ECR${NC}"
echo ""
