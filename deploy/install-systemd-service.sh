#!/bin/bash

################################################################################
# 安装 systemd 服务脚本
# 用途：将 NoFn Trading Agent 配置为 systemd 服务，实现开机自启动
################################################################################

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 配置
SERVICE_NAME="nofn-agent"
SERVICE_FILE="nofn-agent.service"
SYSTEMD_DIR="/etc/systemd/system"
CURRENT_USER=$(whoami)

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

print_header "🔧 安装 NoFn Trading Agent Systemd 服务"

# 检查是否有 sudo 权限
if [ "$EUID" -ne 0 ]; then
    print_info "此脚本需要 sudo 权限"
    echo "请使用 sudo 运行:"
    echo "  sudo $0"
    exit 1
fi

# 检查服务文件是否存在
if [ ! -f "$SERVICE_FILE" ]; then
    print_error "服务文件不存在: $SERVICE_FILE"
    exit 1
fi

# 获取实际用户（如果使用 sudo 运行）
if [ -n "$SUDO_USER" ]; then
    ACTUAL_USER="$SUDO_USER"
else
    ACTUAL_USER="$CURRENT_USER"
fi

print_info "当前用户: $ACTUAL_USER"
echo ""

# 1. 复制服务文件
print_info "复制服务文件到 $SYSTEMD_DIR..."
cp "$SERVICE_FILE" "$SYSTEMD_DIR/${SERVICE_NAME}@.service"
print_success "服务文件已复制"
echo ""

# 2. 重新加载 systemd
print_info "重新加载 systemd..."
systemctl daemon-reload
print_success "systemd 已重新加载"
echo ""

# 3. 启用服务（开机自启动）
print_info "启用服务（开机自启动）..."
systemctl enable "${SERVICE_NAME}@${ACTUAL_USER}.service"
print_success "服务已启用"
echo ""

# 4. 启动服务
print_info "启动服务..."
systemctl start "${SERVICE_NAME}@${ACTUAL_USER}.service"
sleep 2
print_success "服务已启动"
echo ""

# 5. 检查服务状态
print_header "📊 服务状态"
systemctl status "${SERVICE_NAME}@${ACTUAL_USER}.service" --no-pager || true
echo ""

# 完成
print_header "✅ 安装完成！"

echo -e "${BLUE}📋 常用服务管理命令:${NC}"
echo ""
echo "  查看服务状态:"
echo "    sudo systemctl status ${SERVICE_NAME}@${ACTUAL_USER}"
echo ""
echo "  启动服务:"
echo "    sudo systemctl start ${SERVICE_NAME}@${ACTUAL_USER}"
echo ""
echo "  停止服务:"
echo "    sudo systemctl stop ${SERVICE_NAME}@${ACTUAL_USER}"
echo ""
echo "  重启服务:"
echo "    sudo systemctl restart ${SERVICE_NAME}@${ACTUAL_USER}"
echo ""
echo "  禁用开机自启动:"
echo "    sudo systemctl disable ${SERVICE_NAME}@${ACTUAL_USER}"
echo ""
echo "  查看服务日志:"
echo "    sudo journalctl -u ${SERVICE_NAME}@${ACTUAL_USER} -f"
echo ""
echo -e "${GREEN}🎉 NoFn Trading Agent 现在会在系统启动时自动运行！${NC}"
echo ""
