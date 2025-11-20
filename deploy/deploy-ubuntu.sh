#!/bin/bash

################################################################################
# Ubuntu 服务器部署脚本
# 用途：在 Ubuntu 服务器上部署和管理 NoFn Trading Agent
################################################################################

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 配置变量
PROJECT_NAME="nofn-trading-agent"
CONTAINER_NAME="nofn-trading-agent"
AWS_REGION="ap-east-1"
AWS_ACCOUNT_ID="736976853365"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPOSITORY="njkj/trading-agent"
IMAGE_TAG="${1:-latest}"  # 默认使用 latest，可通过参数指定
FULL_IMAGE="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

# 工作目录
WORK_DIR="$HOME/nofn-trading-agent"
ENV_FILE="$WORK_DIR/.env"
CONFIG_DIR="$WORK_DIR/config"
LOGS_DIR="$WORK_DIR/logs"
DATA_DIR="$WORK_DIR/data"

################################################################################
# 函数定义
################################################################################

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

# 检查 Docker 是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装"
        echo ""
        echo "请先安装 Docker:"
        echo "  curl -fsSL https://get.docker.com | sh"
        echo "  sudo usermod -aG docker \$USER"
        echo "  newgrp docker"
        exit 1
    fi
    print_success "Docker 已安装"
}

# 检查 AWS CLI 是否安装
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI 未安装"
        echo ""
        echo "请先安装 AWS CLI:"
        echo "  curl \"https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip\" -o \"awscliv2.zip\""
        echo "  unzip awscliv2.zip"
        echo "  sudo ./aws/install"
        exit 1
    fi
    print_success "AWS CLI 已安装"
}

# 创建工作目录
create_directories() {
    print_info "创建工作目录..."
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "$DATA_DIR"
    print_success "工作目录已创建: $WORK_DIR"
}

# 检查环境变量文件
check_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        print_error "环境变量文件不存在: $ENV_FILE"
        echo ""
        echo "请创建 .env 文件并配置以下变量:"
        echo ""
        cat > "$ENV_FILE.example" << 'EOF'
# LLM 配置
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# 交易所配置
HYPERLIQUID_PRIVATE_KEY=your_private_key_here
HYPERLIQUID_WALLET_ADDRESS=your_wallet_address_here

# 运行配置
TZ=Asia/Shanghai
LOG_LEVEL=INFO
EOF
        cat "$ENV_FILE.example"
        echo ""
        echo "示例文件已创建: $ENV_FILE.example"
        echo "请复制并编辑:"
        echo "  cp $ENV_FILE.example $ENV_FILE"
        echo "  nano $ENV_FILE"
        exit 1
    fi
    print_success "环境变量文件已存在"
}

# 登录 AWS ECR
login_aws_ecr() {
    print_info "登录 AWS ECR..."
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
    if [ $? -eq 0 ]; then
        print_success "AWS ECR 登录成功"
    else
        print_error "AWS ECR 登录失败"
        exit 1
    fi
}

# 拉取镜像
pull_image() {
    print_info "拉取镜像: $FULL_IMAGE"
    docker pull "$FULL_IMAGE"
    if [ $? -eq 0 ]; then
        print_success "镜像拉取成功"
    else
        print_error "镜像拉取失败"
        exit 1
    fi
}

# 停止旧容器
stop_old_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_info "停止旧容器..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
        print_success "旧容器已停止并移除"
    fi
}

# 启动容器
start_container() {
    print_info "启动容器..."

    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        --env-file "$ENV_FILE" \
        -v "$CONFIG_DIR:/app/config:ro" \
        -v "$LOGS_DIR:/app/logs" \
        -v "$DATA_DIR:/app/data" \
        --memory="1.5g" \
        --cpus="2" \
        "$FULL_IMAGE"

    if [ $? -eq 0 ]; then
        print_success "容器启动成功"
    else
        print_error "容器启动失败"
        exit 1
    fi
}

# 检查容器状态
check_container_status() {
    print_info "检查容器状态..."
    sleep 3

    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_success "容器运行中"
        echo ""
        docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        return 0
    else
        print_error "容器未运行"
        echo ""
        echo "查看日志:"
        echo "  docker logs ${CONTAINER_NAME}"
        return 1
    fi
}

################################################################################
# 主流程
################################################################################

print_header "🚀 NoFn Trading Agent - Ubuntu 部署脚本"

echo -e "${YELLOW}部署配置:${NC}"
echo "  镜像: $FULL_IMAGE"
echo "  容器名: $CONTAINER_NAME"
echo "  工作目录: $WORK_DIR"
echo ""

# 1. 环境检查
print_header "1️⃣  环境检查"
check_docker
check_aws_cli
echo ""

# 2. 创建目录
print_header "2️⃣  准备工作目录"
create_directories
check_env_file
echo ""

# 3. 登录 AWS ECR
print_header "3️⃣  登录 AWS ECR"
login_aws_ecr
echo ""

# 4. 拉取镜像
print_header "4️⃣  拉取 Docker 镜像"
pull_image
echo ""

# 5. 停止旧容器
print_header "5️⃣  停止旧容器"
stop_old_container
echo ""

# 6. 启动新容器
print_header "6️⃣  启动新容器"
start_container
echo ""

# 7. 检查状态
print_header "7️⃣  检查容器状态"
check_container_status
echo ""

# 部署完成
print_header "✅ 部署完成！"

echo -e "${BLUE}📋 常用命令:${NC}"
echo ""
echo "  查看日志:"
echo "    docker logs -f ${CONTAINER_NAME}"
echo ""
echo "  查看状态:"
echo "    docker ps | grep ${CONTAINER_NAME}"
echo ""
echo "  重启容器:"
echo "    docker restart ${CONTAINER_NAME}"
echo ""
echo "  停止容器:"
echo "    docker stop ${CONTAINER_NAME}"
echo ""
echo "  进入容器:"
echo "    docker exec -it ${CONTAINER_NAME} bash"
echo ""
echo "  查看资源使用:"
echo "    docker stats ${CONTAINER_NAME} --no-stream"
echo ""
