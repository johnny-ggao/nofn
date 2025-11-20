#!/bin/bash

################################################################################
# 多平台 Docker 镜像构建脚本
# 用途：构建支持多个 CPU 架构的 Docker 镜像（amd64 和 arm64）
# 适用场景：需要在不同架构服务器上部署（x86_64 Ubuntu 和 ARM Ubuntu）
################################################################################

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 配置
IMAGE_NAME="nofn-nofn-agent"
AWS_REGION="ap-east-1"
AWS_ACCOUNT_ID="736976853365"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPOSITORY="njkj/trading-agent"
TAG="${1:-latest}"

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header "🏗️  多平台镜像构建"

echo -e "${BLUE}📋 构建配置:${NC}"
echo "  - 镜像名称: ${IMAGE_NAME}"
echo "  - 版本标签: ${TAG}"
echo "  - 目标平台: linux/amd64, linux/arm64"
echo "  - ECR 仓库: ${ECR_REGISTRY}/${ECR_REPOSITORY}"
echo ""

# 1. 检查 Docker 是否支持 buildx
print_info "检查 Docker buildx 支持..."
if ! docker buildx version &> /dev/null; then
    print_error "Docker buildx 不可用"
    echo ""
    echo "请升级到 Docker 19.03 或更高版本"
    exit 1
fi
print_success "Docker buildx 可用"
echo ""

# 2. 创建或使用 buildx 构建器
print_info "设置 buildx 构建器..."
BUILDER_NAME="nofn-builder"

if docker buildx inspect "$BUILDER_NAME" &> /dev/null; then
    print_info "使用已存在的构建器: $BUILDER_NAME"
    docker buildx use "$BUILDER_NAME"
else
    print_info "创建新的构建器: $BUILDER_NAME"
    docker buildx create --name "$BUILDER_NAME" --driver docker-container --use
    docker buildx inspect --bootstrap
fi
print_success "构建器已就绪"
echo ""

# 3. 构建多平台镜像
print_header "🔨 开始构建多平台镜像"

print_info "构建目标:"
echo "  - linux/amd64 (x86_64 Ubuntu 服务器)"
echo "  - linux/arm64 (ARM Ubuntu 服务器)"
echo ""

START_TIME=$(date +%s)

# 构建并加载到本地（用于测试）
print_info "构建镜像到本地..."
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag "${IMAGE_NAME}:${TAG}" \
    --tag "${IMAGE_NAME}:latest" \
    --tag "${ECR_REGISTRY}/${ECR_REPOSITORY}:${TAG}" \
    --tag "${ECR_REGISTRY}/${ECR_REPOSITORY}:latest" \
    --load \
    -f docker/Dockerfile \
    .

if [ $? -eq 0 ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    print_success "构建完成！耗时: ${DURATION}秒"
    echo ""
else
    print_error "构建失败"
    exit 1
fi

# 4. 显示镜像信息
print_header "📊 镜像信息"
docker images | grep -E "${IMAGE_NAME}|REPOSITORY"
echo ""

# 5. 验证多架构支持
print_info "验证多架构支持..."
docker buildx imagetools inspect "${IMAGE_NAME}:${TAG}" 2>/dev/null || \
    echo "注意: 本地镜像可能只包含当前架构"
echo ""

print_header "✅ 构建完成"

echo -e "${BLUE}📋 下一步操作:${NC}"
echo ""
echo "1. 推送到 AWS ECR:"
echo "   ${YELLOW}./scripts/push-multiplatform-to-aws.sh${NC}"
echo ""
echo "2. 或手动推送:"
echo "   ${YELLOW}docker push ${ECR_REGISTRY}/${ECR_REPOSITORY}:${TAG}${NC}"
echo ""
echo "3. 在 Ubuntu 服务器上拉取:"
echo "   ${YELLOW}docker pull ${ECR_REGISTRY}/${ECR_REPOSITORY}:${TAG}${NC}"
echo ""
echo -e "${GREEN}💡 提示: 多架构镜像会自动选择匹配服务器架构的版本${NC}"
echo ""
