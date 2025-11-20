.PHONY: help build build-nc build-multiplatform login-aws push-aws push-multiplatform clean

# 默认目标
help:
	@echo "🐳 NoFn Trading Agent - 镜像构建与推送"
	@echo ""
	@echo "📦 构建镜像:"
	@echo "  make build              - 构建 amd64 镜像 (Ubuntu 服务器)"
	@echo "  make build-nc           - 构建镜像 (不使用缓存)"
	@echo "  make build-multiplatform - 构建多架构镜像 (amd64+arm64)"
	@echo ""
	@echo "☁️  推送到 AWS ECR:"
	@echo "  make login-aws          - 登录 AWS ECR"
	@echo "  make push               - 构建并推送多架构镜像 (推荐)"
	@echo "  make push-aws           - 推送当前镜像到 AWS ECR"
	@echo ""
	@echo "🧹 清理:"
	@echo "  make clean              - 清理本地镜像和构建缓存"
	@echo ""
	@echo "📚 完整流程:"
	@echo "  1. make build           # 构建镜像"
	@echo "  2. make login-aws       # 登录 AWS ECR"
	@echo "  3. make push-aws        # 推送镜像"
	@echo ""
	@echo "  或一键推送:"
	@echo "  make push               # 构建并推送多架构镜像"
	@echo ""
	@echo "🐧 Ubuntu 服务器部署:"
	@echo "  在服务器上运行: ./deploy-ubuntu.sh"

# ================================
# 构建镜像
# ================================

# 构建 amd64 架构镜像（默认）
build:
	@echo "🔨 Building Docker image for amd64..."
	@./scripts/docker-build.sh --amd64

# 构建镜像（不使用缓存）
build-nc:
	@echo "🔨 Building Docker image (no cache)..."
	@./scripts/docker-build.sh --no-cache --amd64

# 构建多平台镜像 (amd64 + arm64)
build-multiplatform:
	@echo "🔨 Building multi-platform Docker image..."
	@./scripts/build-multiplatform.sh

# ================================
# AWS ECR 推送
# ================================

# AWS 配置
AWS_REGION ?= ap-east-1
AWS_ACCOUNT_ID ?= 736976853365
ECR_REGISTRY = $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
ECR_REPOSITORY = njkj/trading-agent

# 登录 AWS ECR
login-aws:
	@echo "🔑 Logging in to AWS ECR..."
	@aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_REGISTRY)
	@echo "✅ Successfully logged in to AWS ECR!"

# 推送当前镜像到 AWS ECR
push-aws:
	@echo "⬆️  Pushing image to AWS ECR..."
	@./scripts/push-to-aws.sh

# 构建并推送多平台镜像（推荐）
push-multiplatform:
	@echo "🚀 Building and pushing multi-platform image to AWS ECR..."
	@./scripts/push-multiplatform-to-aws.sh

# 快捷命令：构建并推送多平台镜像
push: login-aws push-multiplatform

# ================================
# 清理
# ================================

# 清理本地镜像和构建缓存
clean:
	@echo "🧹 Cleaning Docker images and build cache..."
	@docker rmi nofn-nofn-agent 2>/dev/null || true
	@docker rmi $(ECR_REGISTRY)/$(ECR_REPOSITORY) 2>/dev/null || true
	@docker buildx prune -f
	@docker system prune -f
	@echo "✅ Cleanup complete!"
