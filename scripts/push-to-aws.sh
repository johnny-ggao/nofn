#!/bin/bash

################################################################################
# AWS ECR 镜像推送脚本
# 用途：将本地构建的镜像推送到 AWS ECR
################################################################################

set -e  # 遇到错误立即退出

# 配置变量
AWS_REGION="ap-east-1"
AWS_ACCOUNT_ID="736976853365"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPOSITORY="njkj/trading-agent"
LOCAL_IMAGE="nofn-nofn-agent"
LOCAL_TAG="${1:-latest}"  # 默认使用 latest，可通过参数指定

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}🚀 AWS ECR 镜像推送${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 1. 检查本地镜像是否存在
echo -e "${YELLOW}📦 检查本地镜像...${NC}"
if ! docker images | grep -q "${LOCAL_IMAGE}.*${LOCAL_TAG}"; then
    echo -e "${RED}❌ 错误: 本地镜像 ${LOCAL_IMAGE}:${LOCAL_TAG} 不存在${NC}"
    echo -e "${YELLOW}请先运行: make build${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 本地镜像存在: ${LOCAL_IMAGE}:${LOCAL_TAG}${NC}"
echo ""

# 2. 标记镜像
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPOSITORY}"
echo -e "${YELLOW}🏷️  标记镜像...${NC}"
echo -e "   源镜像: ${LOCAL_IMAGE}:${LOCAL_TAG}"
echo -e "   目标镜像: ${ECR_IMAGE}:${LOCAL_TAG}"

docker tag "${LOCAL_IMAGE}:${LOCAL_TAG}" "${ECR_IMAGE}:${LOCAL_TAG}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 镜像标记成功${NC}"
else
    echo -e "${RED}❌ 镜像标记失败${NC}"
    exit 1
fi
echo ""

# 3. 推送镜像
echo -e "${YELLOW}⬆️  推送镜像到 AWS ECR...${NC}"
echo -e "   推送地址: ${ECR_IMAGE}:${LOCAL_TAG}"
echo ""

docker push "${ECR_IMAGE}:${LOCAL_TAG}"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}✅ 推送成功！${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo -e "${BLUE}📋 镜像信息:${NC}"
    echo -e "   Registry: ${ECR_REGISTRY}"
    echo -e "   Repository: ${ECR_REPOSITORY}"
    echo -e "   Tag: ${LOCAL_TAG}"
    echo ""
    echo -e "${BLUE}🔗 完整镜像地址:${NC}"
    echo -e "   ${ECR_IMAGE}:${LOCAL_TAG}"
    echo ""
    echo -e "${BLUE}💡 下一步:${NC}"
    echo -e "   在 AWS 服务器上拉取镜像:"
    echo -e "   ${YELLOW}docker pull ${ECR_IMAGE}:${LOCAL_TAG}${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}================================${NC}"
    echo -e "${RED}❌ 推送失败${NC}"
    echo -e "${RED}================================${NC}"
    echo ""
    echo -e "${YELLOW}💡 可能的原因:${NC}"
    echo -e "   1. 未登录 AWS ECR，请先运行:"
    echo -e "      ${BLUE}aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}${NC}"
    echo -e "   2. AWS 凭证过期，请检查 AWS CLI 配置"
    echo -e "   3. ECR 仓库不存在或无权限"
    echo -e "   4. 网络连接问题"
    echo ""
    exit 1
fi
