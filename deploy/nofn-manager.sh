#!/bin/bash

################################################################################
# NoFn Trading Agent 管理脚本
# 用途：在 Ubuntu 服务器上管理容器（启动、停止、重启、查看状态等）
################################################################################

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 配置变量
CONTAINER_NAME="nofn-trading-agent"
WORK_DIR="$HOME/nofn-trading-agent"
LOGS_DIR="$WORK_DIR/logs"

################################################################################
# 函数定义
################################################################################

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
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

# 显示帮助信息
show_help() {
    echo "NoFn Trading Agent 管理脚本"
    echo ""
    echo "用法: $0 <command>"
    echo ""
    echo "命令:"
    echo "  start      - 启动容器"
    echo "  stop       - 停止容器"
    echo "  restart    - 重启容器"
    echo "  status     - 查看容器状态"
    echo "  logs       - 查看实时日志"
    echo "  logs-tail  - 查看最近100行日志"
    echo "  shell      - 进入容器 shell"
    echo "  stats      - 查看资源使用"
    echo "  update     - 更新镜像并重启"
    echo "  clean      - 停止并删除容器"
    echo "  help       - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start"
    echo "  $0 restart"
    echo "  $0 logs"
}

# 检查容器是否存在
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# 检查容器是否运行
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# 启动容器
start_container() {
    print_header "🚀 启动容器"

    if container_running; then
        print_info "容器已经在运行中"
        show_status
        return 0
    fi

    if container_exists; then
        print_info "启动已存在的容器..."
        docker start "$CONTAINER_NAME"
    else
        print_error "容器不存在，请先运行部署脚本"
        echo ""
        echo "运行部署脚本:"
        echo "  ./deploy-ubuntu.sh"
        return 1
    fi

    sleep 2

    if container_running; then
        print_success "容器启动成功"
        show_status
    else
        print_error "容器启动失败"
        echo ""
        echo "查看日志:"
        echo "  docker logs ${CONTAINER_NAME}"
        return 1
    fi
}

# 停止容器
stop_container() {
    print_header "⏹️  停止容器"

    if ! container_running; then
        print_info "容器未运行"
        return 0
    fi

    print_info "正在停止容器..."
    docker stop "$CONTAINER_NAME"

    if [ $? -eq 0 ]; then
        print_success "容器已停止"
    else
        print_error "停止容器失败"
        return 1
    fi
}

# 重启容器
restart_container() {
    print_header "🔄 重启容器"

    if ! container_exists; then
        print_error "容器不存在"
        return 1
    fi

    print_info "正在重启容器..."
    docker restart "$CONTAINER_NAME"

    sleep 2

    if container_running; then
        print_success "容器重启成功"
        show_status
    else
        print_error "容器重启失败"
        echo ""
        echo "查看日志:"
        echo "  docker logs ${CONTAINER_NAME}"
        return 1
    fi
}

# 查看状态
show_status() {
    print_header "📊 容器状态"

    if ! container_exists; then
        print_error "容器不存在"
        echo ""
        echo "请先运行部署脚本:"
        echo "  ./deploy-ubuntu.sh"
        return 1
    fi

    # 容器基本信息
    echo -e "${BLUE}📦 容器信息:${NC}"
    docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}"
    echo ""

    # 运行状态
    if container_running; then
        print_success "运行状态: 运行中"

        # 资源使用
        echo ""
        echo -e "${BLUE}💻 资源使用:${NC}"
        docker stats "$CONTAINER_NAME" --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

        # 健康状态
        echo ""
        echo -e "${BLUE}❤️  健康状态:${NC}"
        health_status=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "N/A")
        echo "  $health_status"
    else
        print_info "运行状态: 已停止"
    fi

    echo ""
}

# 查看日志
show_logs() {
    print_header "📋 实时日志"

    if ! container_exists; then
        print_error "容器不存在"
        return 1
    fi

    echo "按 Ctrl+C 退出日志查看"
    echo ""
    docker logs -f "$CONTAINER_NAME"
}

# 查看最近日志
show_logs_tail() {
    print_header "📋 最近日志 (100行)"

    if ! container_exists; then
        print_error "容器不存在"
        return 1
    fi

    docker logs --tail=100 "$CONTAINER_NAME"
}

# 进入容器 shell
enter_shell() {
    print_header "💻 进入容器 Shell"

    if ! container_running; then
        print_error "容器未运行"
        return 1
    fi

    echo "在容器内执行命令，输入 'exit' 退出"
    echo ""
    docker exec -it "$CONTAINER_NAME" bash
}

# 查看资源使用
show_stats() {
    print_header "💻 资源使用情况"

    if ! container_running; then
        print_error "容器未运行"
        return 1
    fi

    echo "按 Ctrl+C 退出"
    echo ""
    docker stats "$CONTAINER_NAME"
}

# 更新镜像并重启
update_container() {
    print_header "🔄 更新容器"

    # 获取当前镜像信息
    if container_exists; then
        CURRENT_IMAGE=$(docker inspect --format='{{.Config.Image}}' "$CONTAINER_NAME")
        echo -e "${BLUE}当前镜像: ${CURRENT_IMAGE}${NC}"
        echo ""
    else
        print_error "容器不存在，请先运行部署脚本"
        return 1
    fi

    print_info "拉取最新镜像..."
    docker pull "$CURRENT_IMAGE"

    if [ $? -eq 0 ]; then
        print_success "镜像更新成功"
        echo ""
        print_info "重启容器以应用更新..."
        restart_container
    else
        print_error "镜像更新失败"
        return 1
    fi
}

# 清理容器
clean_container() {
    print_header "🧹 清理容器"

    if ! container_exists; then
        print_info "容器不存在，无需清理"
        return 0
    fi

    echo -e "${YELLOW}警告: 这将停止并删除容器（配置和日志不会被删除）${NC}"
    read -p "确认继续? [y/N] " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "操作已取消"
        return 0
    fi

    if container_running; then
        print_info "停止容器..."
        docker stop "$CONTAINER_NAME"
    fi

    print_info "删除容器..."
    docker rm "$CONTAINER_NAME"

    print_success "容器已清理"
}

################################################################################
# 主流程
################################################################################

# 检查参数
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# 执行命令
case "$1" in
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        restart_container
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    logs-tail)
        show_logs_tail
        ;;
    shell)
        enter_shell
        ;;
    stats)
        show_stats
        ;;
    update)
        update_container
        ;;
    clean)
        clean_container
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "未知命令: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
