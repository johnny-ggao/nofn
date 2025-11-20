#!/bin/bash

################################################################################
# 多平台镜像推送到 AWS ECR 脚本
# 用途：构建并推送支持多架构的镜像到 AWS ECR
################################################################################

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 配置
AWS_REGION="ap-east-1"
AWS_ACCOUNT_ID="736976853365"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPOSITORY="njkj/trading-agent"
TAG="${1:-latest}"
FULL_IMAGE="${ECR_REGISTRY}/${ECR_REPOSITORY}:${TAG}"

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

print_header "🚀 多平台镜像推送到 AWS ECR"

echo -e "${BLUE}📋 推送配置:${NC}"
echo "  - 目标仓库: ${ECR_REGISTRY}/${ECR_REPOSITORY}"
echo "  - 版本标签: ${TAG}"
echo "  - 支持架构: linux/amd64, linux/arm64"
echo ""

# 1. 检查 Docker buildx
print_info "检查 Docker buildx..."
if ! docker buildx version &> /dev/null; then
    print_error "Docker buildx 不可用"
    exit 1
fi
print_success "Docker buildx 可用"
echo ""

# 2. 登录 AWS ECR
print_info "登录 AWS ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_REGISTRY"

if [ $? -eq 0 ]; then
    print_success "AWS ECR 登录成功"
else
    print_error "AWS ECR 登录失败"
    exit 1
fi
echo ""

# 3. 设置构建器
print_info "设置构建器..."
BUILDER_NAME="nofn-builder"

if docker buildx inspect "$BUILDER_NAME" &> /dev/null; then
    docker buildx use "$BUILDER_NAME"
else
    docker buildx create --name "$BUILDER_NAME" --driver docker-container --use
    docker buildx inspect --bootstrap
fi
print_success "构建器已就绪"
echo ""

# 4. 构建并推送多平台镜像
print_header "🔨 构建并推送多平台镜像"

print_info "构建平台: linux/amd64, linux/arm64"
echo ""

START_TIME=$(date +%s)

docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag "${FULL_IMAGE}" \
    --tag "${ECR_REGISTRY}/${ECR_REPOSITORY}:latest" \
    --push \
    -f docker/Dockerfile \
    .

if [ $? -eq 0 ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    print_header "✅ 推送成功！"
    echo -e "${GREEN}耗时: ${DURATION}秒${NC}"
    echo ""
else
    echo ""
    print_error "推送失败"
    exit 1
fi

# 5. 验证推送结果
print_header "🔍 验证镜像"

print_info "查询镜像清单..."
docker buildx imagetools inspect "${FULL_IMAGE}"
echo ""

print_header "📊 推送完成"

echo -e "${BLUE}📋 镜像信息:${NC}"
echo "  - 仓库: ${ECR_REGISTRY}/${ECR_REPOSITORY}"
echo "  - 标签: ${TAG}"
echo "  - 架构: amd64, arm64"
echo ""

echo -e "${BLUE}💡 在 Ubuntu 服务器上拉取:${NC}"
echo "  ${YELLOW}docker pull ${FULL_IMAGE}${NC}"
echo ""

echo -e "${BLUE}💡 在服务器上部署:${NC}"
echo "  ${YELLOW}./deploy-ubuntu.sh ${TAG}${NC}"
echo ""

echo -e "${GREEN}🎉 多平台镜像已成功推送到 AWS ECR！${NC}"
echo -e "${GREEN}   服务器会自动选择匹配的架构版本${NC}"
echo ""
