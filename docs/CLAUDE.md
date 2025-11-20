# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

**nofn** 是一个基于 LangChain 构建的交易智能体系统，能够通过 LLM 驱动执行加密货币交易操作。该系统使用 Hansen 策略进行自主量化交易，通过 CCXT 在多个交易所执行交易操作（开仓/平仓、设置止盈止损）。

### 核心技术栈
- **LangChain**: LLM 集成框架，用于管理交易工作流和决策
- **CCXT**: 统一的加密货币交易所 API（初期接入 Hyperliquid，设计上支持 Binance、OKX 等）
- **Python 3.12+**: 运行时环境
- **uv**: 包管理器和虚拟环境工具

## 开发环境

### 包管理
本项目使用 `uv` 进行依赖管理。常用命令：

```bash
# 安装依赖
uv sync

# 添加新依赖
uv add <package-name>

# 添加开发依赖
uv add --dev <package-name>

# 运行 Python 脚本
uv run python <script.py>

# 手动激活虚拟环境
source .venv/bin/activate  # macOS/Linux
```

### 核心依赖
项目启动时需要安装：
- `langchain` 和 `langchain-core` - LLM 集成和工作流
- `langchain-openai` 和 `langchain-anthropic` - LLM 提供商集成
- `ccxt` - 交易所连接
- `python-dotenv` - 环境变量管理（用于 API 密钥）
- `numpy` - 技术指标计算

## 架构设计

### 高层架构

系统采用**线性工作流架构**，基于 Hansen 策略执行以下步骤：

1. **市场数据获取 (fetch_market_data)**: 并行获取账户余额、持仓和多时间框架K线数据
2. **市场分析 (analyze_market)**: 计算技术指标（EMA、MACD、RSI、ATR、布林带、OBV等）
3. **信号生成 (generate_signals)**: 使用 LangChain Chain（prompt | llm）让 LLM 分析市场并生成交易信号
4. **风险评估 (assess_risk)**: 评估交易信号的风险（置信度、回撤、仓位限制等）
5. **交易执行 (execute_trades)**: 通过 CCXT 执行通过风控的交易
6. **持仓监控 (monitor_positions)**: 监控活跃持仓的盈亏和止盈止损
7. **策略优化 (update_strategy)**: 清理临时数据并根据绩效优化策略参数

### 状态管理

系统使用 **JSON 文件持久化**状态，包含：
- `session_id`: 会话标识
- `market_data`: 市场数据（余额、持仓、K线）
- `technical_indicators`: 各时间框架的技术指标
- `trading_signals`: LLM 生成的交易信号列表
- `risk_assessment`: 风险评估结果（approved/rejected/adjustable）
- `execution_results`: 交易执行结果
- `active_positions`: 活跃持仓列表
- `error_log`: 错误日志

### 交易所适配器模式

每个交易所适配器应实现统一接口：

```python
class BaseExchangeAdapter(ABC):
    @abstractmethod
    async def open_position(self, symbol, side, amount, params): ...
    @abstractmethod
    async def close_position(self, symbol, position_id, params): ...
    @abstractmethod
    async def set_stop_loss_take_profit(self, symbol, position_id, sl, tp): ...
    @abstractmethod
    async def get_positions(self, symbol=None): ...
```

这确保了添加新交易所时无需修改核心工作流逻辑。

### 配置策略

使用配置驱动的方式选择交易所：
- 使用环境变量存储 API 凭证（`.env` 文件，永不提交到版本控制）
- 配置文件（`config.yaml` 或类似）存储交易所偏好和默认参数
- 运行时根据用户提示词或配置选择交易所

### LLM 集成要点

LLM 应提取以下信息：
- **动作类型**: open_long, open_short, close_position, modify_sl_tp, cancel_order
- **交易对**: 标准格式的交易对
- **数量**: 仓位大小（含单位 - 合约数、美元价值等）
- **订单类型**: market（市价）, limit（限价）
- **价格**: 用于限价单
- **止损**: 价格水平或百分比
- **止盈**: 价格水平或百分比
- **杠杆**: 如适用

使用结构化输出（JSON 模式或函数调用）确保可靠解析。

### 错误处理策略

实现全面的错误处理：
- LLM 提取失败 → 请求用户澄清
- 交易所 API 错误 → 返回用户友好的错误消息和建议
- 网络/超时错误 → 实现指数退避的重试逻辑
- 余额不足 → 执行前警告用户
- 无效参数 → 发送到交易所前进行验证

### 安全考虑

- **永不提交 API 密钥**: 使用 `.env` 存储凭证，添加到 `.gitignore`
- **验证所有 LLM 输出**: 永不直接信任 LLM 输出用于交易操作
- **实现确认流程**: 对于大额订单或高风险操作，要求用户明确确认
- **速率限制**: 遵守交易所 API 速率限制
- **模拟交易模式**: 实现测试模式，无需真实资金

## 推荐项目结构

```
nofn/
├── src/
│   ├── agents/          # 交易智能体实现
│   │   └── hansen/      # Hansen 策略智能体
│   │       ├── agent.py             # 主智能体（简化版，JSON 状态）
│   │       ├── langchain_agent.py   # LangChain 标准实现
│   │       ├── market_analyzer.py   # 市场分析
│   │       ├── trading_executor.py  # 交易执行
│   │       └── ...
│   ├── adapters/        # 交易所适配器实现
│   │   ├── base.py              # 基础接口
│   │   └── hyperliquid.py       # Hyperliquid 实现
│   ├── models/          # Pydantic 模型
│   │   └── strategy.py          # 策略相关模型
│   ├── prompts/         # LLM 提示词模板
│   │   └── hansen.txt           # Hansen 策略提示词
│   └── utils/           # 辅助函数
│       ├── indicators.py        # 技术指标计算
│       ├── config.py            # 配置管理
│       └── context_manager.py   # 交易上下文管理
├── config/              # 配置文件
│   └── config.yaml
├── .env                 # 环境变量（不提交）
├── main.py              # 统一入口点
└── pyproject.toml       # 项目元数据和依赖
```

## 测试方法

- 使用 `pytest` 进行单元测试和集成测试
- 单元测试中 Mock CCXT 交易所调用
- 集成测试使用 CCXT 沙箱/测试网环境
- 使用测试模式验证工作流（`main.py test`）
- 使用各种市场条件验证 LLM 输出解析

## 核心实现要点

1. **依赖管理**: 使用 uv 管理项目依赖（langchain, ccxt, numpy等）
2. **交易所适配器**: 基于 CCXT 实现统一的交易所接口
3. **Hansen 策略**: 实现基于夏普比率最大化的交易策略
4. **LangChain 集成**: 使用 ChatPromptTemplate 和 Chain（prompt | llm）进行 LLM 决策
5. **状态持久化**: 使用 JSON 文件保存和恢复交易状态
6. **技术分析**: 实现多时间框架（1m/15m/1h/4h）的技术指标计算
7. **风险管理**: 实现严格的风控检查（夏普比率、置信度、仓位限制）
8. **错误处理**: 实现指数退避的重试机制和详细错误日志
9. **配置管理**: 使用 YAML 和环境变量管理配置
10. **Docker 部署**: 支持跨平台 Docker 镜像构建和 AWS ECR 部署

## 运行模式

系统支持三种运行模式：

1. **策略模式** (`main.py strategy`): 自主循环运行，每 3 分钟执行完整工作流
2. **测试模式** (`main.py test`): 单次迭代测试，查看完整分析，不执行交易
3. **交互模式** (`main.py interactive`): 通过自然语言手动交易

## Hansen 策略核心

- **目标**: 最大化夏普比率（Sharpe Ratio = 平均收益 / 收益波动率）
- **质量优于数量**: 只在高质量信号时交易（信号强度 >= 80/100）
- **严格风控**: 止损强制要求，风险回报比 >= 1:3
- **自适应管理**: 根据夏普比率动态调整策略激进程度