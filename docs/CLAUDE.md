# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**NOFN** 是一个基于 LLM 的智能量化交易系统，采用模块化架构设计。通过 CCXT 支持多个交易所（Binance、OKX、Bybit、Gate、KuCoin、MEXC、Bitget、Hyperliquid），使用 LangChain/LangGraph 进行 LLM 决策。

### 核心技术栈
- **Python 3.12+** + **uv** 包管理
- **LangChain/LangGraph**: LLM 集成与工作流编排
- **CCXT**: 统一交易所 API
- **SQLAlchemy + SQLite**: 状态持久化
- **LanceDB**: 向量存储（长期记忆）
- **Pydantic**: 数据模型验证

## 常用命令

```bash
# 安装依赖
uv sync

# 运行主程序
uv run python main.py

# 运行测试
uv run pytest tests/ -v

# 运行单个测试文件
uv run pytest tests/test_new_architecture.py -v

# 类型检查
uv run mypy src/

# 代码格式化
uv run black src/ tests/
uv run ruff check src/ tests/
```

## 架构设计

### 核心流程

```
StrategyAgent (src/strategy/agent.py)
    │
    ├── StrategyRuntime (src/trading/engine.py)
    │       │
    │       └── GraphDecisionCoordinator (基于 LangGraph)
    │               │
    │               ├── 1. 同步仓位 (portfolio_service)
    │               ├── 2. 构建特征 (features_pipeline)
    │               ├── 3. LangGraph 工作流:
    │               │       reflect → decide → execute → record → summarize
    │               ├── 4. 状态自动持久化 (SqliteSaver checkpointer)
    │               └── 5. 短期记忆由 LangGraph State 自动管理
    │
    └── 循环运行 (decide_interval 秒)
```

### 模块职责

| 模块路径 | 职责 |
|---------|------|
| `src/strategy/agent.py` | 策略代理，编排交易循环，支持反思模式 |
| `src/trading/engine.py` | 交易引擎：`StrategyRuntime` + `DecisionCoordinator` |
| `src/trading/market/` | 市场数据获取与特征计算（K线、MTF指标） |
| `src/trading/decision/` | LLM 决策器：`LlmComposer`、系统提示词 |
| `src/trading/execution/` | 订单执行：`ccxt_trading.py`（实盘）、`paper_trading.py`（模拟） |
| `src/trading/portfolio/` | 仓位与资金管理 |
| `src/trading/history/` | 交易历史与摘要（Digest） |
| `src/trading/reflection/` | 反思分析与策略调整 |
| `src/trading/graph/` | LangGraph 工作流、状态管理、记忆层（短期+长期） |
| `src/trading/db/` | SQLAlchemy 持久化层 |
| `src/trading/templates/` | 策略提示词模板 |

### 关键数据流

1. **特征管道** (`market/pipeline.py`):
   - `DataSourceFeatureComputer` → 从交易所获取 K 线
   - `CandleBasedFeatureComputer` → 计算技术指标（EMA、MACD、RSI、ADX、ATR、BB、OBV）
   - `MarketSnapshotComputer` → 实时行情快照

2. **决策器** (`decision/llm_composer.py`):
   - 构建系统提示词（含策略模板 + 特征 + 仓位 + 历史摘要）
   - 调用 LLM 生成 `TradeInstruction[]`
   - 支持多 LLM 提供商：OpenAI、OpenRouter、DeepSeek、Anthropic、DashScope

3. **执行网关** (`execution/`):
   - `CcxtExecutionGateway`: 实盘交易（支持市价单、止盈止损）
   - `PaperTradingGateway`: 模拟交易

### 配置系统

采用 **YAML + 环境变量** 混合配置：
- `config/config.yaml`: 交易所、策略、LLM 等配置
- `.env`: API 密钥等敏感信息

配置加载入口：`src/config.py` → `Settings` 类

### 策略模板

位于 `src/trading/templates/`：
- `default.txt`: 稳健顺势（置信度 > 0.7）
- `aggressive.txt`: 激进动量（置信度 > 0.6）
- `insane.txt`: 极端激进（置信度 > 0.5）
- `funding_rate.txt`: 资金费率套利
- `nof1.txt`: 自定义策略

### 技术指标（MTF 多时间框架）

**1H 周期**: EMA(7,21,55), MACD(6,13,5), RSI(14), ADX(14), ATR(14), BB(20,2)
**15M 周期**: EMA(8,21,50), MACD(6,13,5), RSI(14), Stochastic(14,3,3), Volume ROC
**5M 周期**: EMA(8), RSI(9), MACD(5,10,3), Volume MA

### 反思模式

启用后（`enable_reflection=True`）：
- 分析历史表现（夏普比、胜率、回撤）
- 识别问题模式（过度交易、连续亏损）
- 动态调整置信度阈值
- 严重问题时触发冷静期

## 数据模型

核心模型定义在 `src/trading/models.py`：
- `UserRequest`: 用户请求（LLM配置 + 交易所配置 + 交易配置）
- `TradeInstruction`: 交易指令
- `TradeHistoryEntry`: 交易记录
- `FeatureVector`: 特征向量
- `ComposeContext`: 决策上下文
- `DecisionCycleResult`: 决策周期结果

## 数据库模型

位于 `src/trading/db/models/`：
- `StrategyModel`: 策略记录
- `StrategyDetailModel`: 策略详情/交易记录
- `StrategyHoldingModel`: 持仓快照

## 记忆管理（LangGraph 原生）

短期记忆现在由 **LangGraph State + Checkpointer** 自动管理，无需手动维护。

### 记忆层次

| 层级 | 存储 | 管理方式 |
|-----|------|---------|
| 短期记忆 | `TradingState.memories` | LangGraph State reducer 自动合并，保留最近 10 条 |
| 中期摘要 | `TradingState.summaries` | 当 memories 满时自动压缩，保留最近 5 个摘要 |
| 长期记忆 | LangGraph Store + LanceDB | 跨会话持久化，支持向量搜索 |

### 核心类型

定义在 `src/trading/graph/state.py`：
- `TradingState`: LangGraph 状态（含 memories、summaries、pending_signals）
- `DecisionMemory`: 单条决策记忆
- `DecisionSummary`: 压缩的历史摘要

### Checkpointer 配置

每个策略有独立的 SQLite checkpoint 文件：
```
data/langgraph_{strategy_id}.db
```

状态恢复：
- 策略重启时自动从 checkpoint 恢复
- `thread_id` = `strategy_id`