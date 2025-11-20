# Ubuntu 服务器部署指南

本指南介绍如何在 Ubuntu 服务器上部署和管理 NoFn Trading Agent。

## 📋 目录

- [前置要求](#前置要求)
- [快速开始](#快速开始)
- [部署流程](#部署流程)
- [日常管理](#日常管理)
- [开机自启动](#开机自启动)
- [故障排查](#故障排查)

---

## 前置要求

### 1. 系统要求
- Ubuntu 20.04+ (推荐 22.04 LTS)
- 至少 2GB RAM
- 至少 10GB 可用磁盘空间

### 2. 需要安装的软件

#### 安装 Docker
```bash
# 下载并安装 Docker
curl -fsSL https://get.docker.com | sh

# 将当前用户添加到 docker 组
sudo usermod -aG docker $USER

# 重新加载组权限
newgrp docker

# 验证安装
docker --version
```

#### 安装 AWS CLI
```bash
# 下载 AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

# 解压
unzip awscliv2.zip

# 安装
sudo ./aws/install

# 验证安装
aws --version

# 配置 AWS 凭证
aws configure
```

配置 AWS 凭证时需要输入：
- AWS Access Key ID
- AWS Secret Access Key
- Default region name: `ap-east-1`
- Default output format: `json`

---

## 快速开始

### 1. 下载部署脚本

将以下文件上传到服务器：
```
deploy-ubuntu.sh          # 部署脚本
nofn-manager.sh          # 管理脚本
nofn-agent.service       # systemd 服务文件（可选）
install-systemd-service.sh  # systemd 安装脚本（可选）
```

或者使用 git 克隆：
```bash
# 如果项目在 git 仓库中
git clone <your-repo-url>
cd nofn
```

### 2. 添加执行权限
```bash
chmod +x deploy-ubuntu.sh
chmod +x nofn-manager.sh
chmod +x install-systemd-service.sh
```

### 3. 首次部署
```bash
./deploy-ubuntu.sh
```

首次运行会创建必要的目录和配置文件模板。按照提示编辑配置文件后，再次运行部署脚本。

---

## 部署流程

### 步骤 1: 运行部署脚本

```bash
./deploy-ubuntu.sh
```

**部署脚本会自动执行以下操作**:
1. ✅ 检查 Docker 和 AWS CLI 是否安装
2. ✅ 创建工作目录 (`~/nofn-trading-agent`)
3. ✅ 检查环境变量文件 (`.env`)
4. ✅ 登录 AWS ECR
5. ✅ 拉取最新镜像
6. ✅ 停止旧容器（如果存在）
7. ✅ 启动新容器
8. ✅ 检查容器状态

### 步骤 2: 配置环境变量

如果是首次部署，需要配置 `.env` 文件：

```bash
cd ~/nofn-trading-agent
nano .env
```

填入必要的配置：
```bash
# LLM 配置
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxx...
OPENAI_API_KEY=sk-xxx...

# 交易所配置
HYPERLIQUID_PRIVATE_KEY=0x...
HYPERLIQUID_WALLET_ADDRESS=0x...

# 运行配置
TZ=Asia/Shanghai
LOG_LEVEL=INFO
```

保存后再次运行部署脚本：
```bash
./deploy-ubuntu.sh
```

### 步骤 3: 验证部署

```bash
# 检查容器是否运行
docker ps | grep nofn

# 查看日志
docker logs -f nofn-trading-agent
```

---

## 日常管理

使用 `nofn-manager.sh` 脚本进行日常管理：

### 查看帮助
```bash
./nofn-manager.sh help
```

### 启动容器
```bash
./nofn-manager.sh start
```

### 停止容器
```bash
./nofn-manager.sh stop
```

### 重启容器
```bash
./nofn-manager.sh restart
```

### 查看状态
```bash
./nofn-manager.sh status
```

输出示例：
```
================================
📊 容器状态
================================

📦 容器信息:
CONTAINER ID   NAMES                STATUS         IMAGE
abc123def456   nofn-trading-agent   Up 2 hours     736976853365.dkr.ecr.ap-east-1.amazonaws.com/njkj/trading-agent:latest

✅ 运行状态: 运行中

💻 资源使用:
CONTAINER          CPU %     MEM USAGE / LIMIT     MEM %
nofn-trading-agent 5.23%     512MiB / 1.5GiB       34.13%

❤️  健康状态:
  healthy
```

### 查看日志
```bash
# 实时日志（按 Ctrl+C 退出）
./nofn-manager.sh logs

# 查看最近100行
./nofn-manager.sh logs-tail
```

### 进入容器 Shell
```bash
./nofn-manager.sh shell
```

### 查看资源使用
```bash
./nofn-manager.sh stats
```

### 更新镜像并重启
```bash
./nofn-manager.sh update
```

这会：
1. 拉取最新镜像
2. 重启容器以应用更新

### 清理容器
```bash
./nofn-manager.sh clean
```

这会停止并删除容器（不会删除配置和日志）。

---

## 开机自启动

使用 systemd 服务实现开机自启动。

### 安装 systemd 服务

```bash
sudo ./install-systemd-service.sh
```

安装后，容器会在系统启动时自动运行。

### Systemd 服务管理

```bash
# 查看服务状态
sudo systemctl status nofn-agent@$USER

# 启动服务
sudo systemctl start nofn-agent@$USER

# 停止服务
sudo systemctl stop nofn-agent@$USER

# 重启服务
sudo systemctl restart nofn-agent@$USER

# 禁用开机自启动
sudo systemctl disable nofn-agent@$USER

# 启用开机自启动
sudo systemctl enable nofn-agent@$USER

# 查看服务日志
sudo journalctl -u nofn-agent@$USER -f
```

### 卸载 systemd 服务

```bash
# 停止并禁用服务
sudo systemctl stop nofn-agent@$USER
sudo systemctl disable nofn-agent@$USER

# 删除服务文件
sudo rm /etc/systemd/system/nofn-agent@.service

# 重新加载 systemd
sudo systemctl daemon-reload
```

---

## 故障排查

### 问题 1: 容器无法启动

**现象**:
```bash
./nofn-manager.sh status
❌ 容器不存在
```

**解决**:
```bash
# 重新部署
./deploy-ubuntu.sh
```

### 问题 2: 容器频繁重启

**检查日志**:
```bash
docker logs nofn-trading-agent
```

**常见原因**:
- 配置错误（检查 `.env` 文件）
- API Key 无效
- 网络连接问题

### 问题 3: 无法拉取镜像

**现象**:
```bash
Error response from daemon: Get "https://736976853365.dkr.ecr.ap-east-1.amazonaws.com/v2/": unauthorized
```

**解决**:
```bash
# 重新登录 AWS ECR
aws ecr get-login-password --region ap-east-1 | docker login --username AWS --password-stdin 736976853365.dkr.ecr.ap-east-1.amazonaws.com

# 重新部署
./deploy-ubuntu.sh
```

### 问题 4: 内存不足

**现象**:
容器经常被 OOM Killer 杀掉

**解决**:
编辑 `deploy-ubuntu.sh`，调整内存限制：
```bash
# 在 start_container 函数中修改
--memory="2g"  # 增加到 2GB
```

### 问题 5: 端口占用

**现象**:
```
Error starting userland proxy: listen tcp4 0.0.0.0:8000: bind: address already in use
```

**解决**:
```bash
# 查找占用端口的进程
sudo lsof -i :8000

# 停止占用端口的进程
sudo kill <PID>
```

### 问题 6: 磁盘空间不足

**检查磁盘空间**:
```bash
df -h
```

**清理 Docker 资源**:
```bash
# 清理未使用的镜像和容器
docker system prune -a

# 清理旧日志
cd ~/nofn-trading-agent/logs
rm *.log.old
```

---

## 文件结构

部署后的文件结构：

```
~/nofn-trading-agent/
├── .env                    # 环境变量配置（敏感信息）
├── .env.example            # 环境变量示例
├── config/                 # 配置文件目录（从镜像挂载）
├── logs/                   # 日志文件目录
│   ├── agent.log
│   └── error.log
└── data/                   # 数据目录（预留）
```

---

## 监控建议

### 1. 日志监控
```bash
# 使用 tail 监控日志
tail -f ~/nofn-trading-agent/logs/agent.log

# 或使用 Docker 日志
docker logs -f nofn-trading-agent
```

### 2. 资源监控
```bash
# 实时监控
watch -n 5 './nofn-manager.sh status'

# 或使用 Docker stats
docker stats nofn-trading-agent
```

### 3. 告警设置
建议配置以下告警：
- 容器停止运行
- 内存使用超过 80%
- CPU 使用超过 80%
- 磁盘空间低于 20%

---

## 备份和恢复

### 备份配置和日志
```bash
# 创建备份
cd ~
tar -czf nofn-backup-$(date +%Y%m%d).tar.gz nofn-trading-agent/.env nofn-trading-agent/logs/

# 备份到其他服务器
scp nofn-backup-*.tar.gz user@backup-server:/backups/
```

### 恢复
```bash
# 解压备份
tar -xzf nofn-backup-20251119.tar.gz

# 重新部署
./deploy-ubuntu.sh
```

---

## 安全建议

1. **保护 .env 文件**
   ```bash
   chmod 600 ~/nofn-trading-agent/.env
   ```

2. **定期更新镜像**
   ```bash
   ./nofn-manager.sh update
   ```

3. **定期审查日志**
   ```bash
   ./nofn-manager.sh logs-tail
   ```

4. **使用防火墙**
   ```bash
   sudo ufw enable
   sudo ufw allow 22/tcp  # SSH
   ```

5. **启用自动安全更新**
   ```bash
   sudo apt install unattended-upgrades
   sudo dpkg-reconfigure -plow unattended-upgrades
   ```

---

## 性能优化

### 1. 调整 Docker 资源限制
编辑 `deploy-ubuntu.sh` 中的资源配置：
```bash
--memory="2g"      # 内存限制
--cpus="2"         # CPU 限制
```

### 2. 优化日志滚动
配置日志滚动以防止日志文件过大：
```bash
# 创建 logrotate 配置
sudo nano /etc/logrotate.d/nofn-agent

# 内容：
/home/*/nofn-trading-agent/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

---

## 常用运维场景

### 场景 1: 首次部署
```bash
chmod +x deploy-ubuntu.sh nofn-manager.sh
./deploy-ubuntu.sh
# 按提示配置 .env
./deploy-ubuntu.sh  # 再次运行
```

### 场景 2: 更新应用
```bash
./nofn-manager.sh update
```

### 场景 3: 查看最近发生了什么
```bash
./nofn-manager.sh status
./nofn-manager.sh logs-tail
```

### 场景 4: 应用出问题了
```bash
./nofn-manager.sh restart
docker logs nofn-trading-agent
```

### 场景 5: 服务器重启后检查
```bash
./nofn-manager.sh status
```

如果安装了 systemd 服务，容器会自动启动。

---

## 总结

**部署流程**:
1. ✅ 安装 Docker 和 AWS CLI
2. ✅ 运行 `./deploy-ubuntu.sh`
3. ✅ 配置 `.env` 文件
4. ✅ 再次运行 `./deploy-ubuntu.sh`
5. ✅ （可选）安装 systemd 服务

**日常管理**:
- 使用 `./nofn-manager.sh` 管理容器
- 使用 systemd 命令管理服务（如已安装）

**监控**:
- 定期查看状态和日志
- 关注资源使用情况

有问题？查看 [故障排查](#故障排查) 章节。
