# NoFn - Hansen 策略自主交易系统

基于 LangChain 构建的智能交易系统，使用 **Hansen 策略**实现自主量化交易。

## ✨ 核心特性

### 🎯 Hansen 策略驱动
- **最大化夏普比率** - 质量优于数量
- **严格风控** - 止损强制，风险回报 >= 1:3
- **自适应管理** - 根据绩效动态调整策略
- **高质量信号** - 只在确定性高时交易

### 📊 完整技术分析
- **多时间框架** - 1m/15m/1h/4h 全面分析
- **10+ 技术指标** - EMA、MACD、RSI、ATR、BB、OBV 等
- **智能评分** - 加权信号强度（0-100）
- **6 维验证** - 多角度确认交易机会

### 🤖 LLM 增强决策
- **Chain-of-Thought** - 完整推理过程
- **自然语言** - 清晰可解释的决策
- **结构化输出** - JSON 格式交易信号
- **多 LLM 支持** - DeepSeek、OpenAI、自定义

### 🛡️ 风险管理
- **夏普比率监控** - 实时绩效追踪
- **过度交易保护** - 自动频率控制
- **动态杠杆** - 基于波动率调整
- **回撤保护** - 自动止损和减仓

## 🚀 快速开始

### 1. 安装

```bash
# 克隆项目
git clone <repo-url>
cd nofn

# 安装依赖
uv sync
```

### 2. 配置

```bash
# 创建 .env 文件
cat > .env << EOF
# LLM API Key
DEEPSEEK_API_KEY=your-deepseek-api-key

# 交易所 API Keys
HYPERLIQUID_API_KEY=your-hyperliquid-key
HYPERLIQUID_API_SECRET=your-hyperliquid-secret
EOF
```

### 3. 配置 Agent 模式

编辑 `config/config.yaml`：

```yaml
# Agent 配置
agent:
  # Agent 模式：
  # - "tools": 工具调用模式（推荐）
  # - "langchain": 线性工作流模式
  mode: "tools"

  # 只读模式（仅对 tools 模式有效）
  # - true: 只读，不执行真实交易（安全）
  # - false: 完整交易模式（谨慎！）
  read_only: true
```

### 4. 运行

```bash
# 直接运行（使用配置文件中的设置）
uv run python main.py
```

**修改运行模式**：
- 编辑 `config/config.yaml` 中的 `agent.mode` 和 `agent.read_only`
- 无需命令行参数，配置更清晰

## 📖 运行模式

### 工具调用模式 ⭐ 推荐（默认）

LLM 自主决定调用哪些工具进行分析和交易。

**配置**（在 `config/config.yaml` 中）：
```yaml
agent:
  mode: "tools"
  read_only: true  # 只读模式（安全）
```

**特点**：
- ✅ LLM 自主决策，灵活调用工具
- ✅ 15 个专业工具（查询、分析、交易、性能）
- ✅ 默认只读模式，安全测试
- ✅ 强制止损、风险回报比检查
- ✅ 更符合 LangChain 设计理念

**可用工具**：
- 查询类：余额、持仓、价格、K线
- 分析类：技术指标、趋势分析、风险回报比、仓位计算
- 交易类：开多/空仓、平仓、止损止盈
- 性能类：绩效统计、交易记录

**启用完整交易**：
```yaml
agent:
  mode: "tools"
  read_only: false  # ⚠️ 会执行真实交易
```

### 线性工作流模式

按固定 7 步流程执行（传统方式）。

**配置**（在 `config/config.yaml` 中）：
```yaml
agent:
  mode: "langchain"
  read_only: true  # 此选项对 langchain 模式无效
```

**特点**：
- ✅ 固定 7 步流程（数据→分析→信号→风控→执行→监控→优化）
- ✅ 执行过程可预测
- ✅ 适合固定策略流程

**适用场景**：
- 需要可预测的执行流程
- 调试特定步骤
- 作为 Tool-based 模式的备选方案

## 技术栈

- **LangChain** - LLM 集成和工作流
- **CCXT** - 交易所 API
- **NumPy** - 技术指标计算
- **Pydantic** - 数据验证
- **Python 3.12+** - 运行环境
- **uv** - 包管理

## 📁 项目结构

```
nofn/
├── main.py                    # 统一入口点 ⭐
├── src/
│   ├── agents/
│   │   ├── strategy/          # 策略节点
│   │   │   ├── market_analyzer.py      # 市场分析
│   │   │   ├── performance_tracker.py  # 绩效追踪
│   │   │   └── strategy_decision.py    # 策略决策
│   │   ├── trading/           # 交易节点
│   │   └── risk/              # 风控节点
│   ├── core/
│   │   └── graph/
│   │       ├── strategy_graph.py       # 策略工作流 ⭐
│   │       └── trading_graph.py        # 交互工作流
│   ├── adapters/              # 交易所适配器
│   │   ├── hyperliquid.py    # Hyperliquid
│   │   └── base.py           # 基类
│   ├── models/
│   │   ├── strategy.py       # 策略模型 ⭐
│   │   └── trading.py        # 交易模型
│   ├── utils/
│   │   ├── indicators.py     # 技术指标 ⭐
│   │   ├── llm_config.py     # LLM 配置
│   │   └── config.py         # 系统配置
│   └── prompts/
│       └── hansen.txt        # Hansen 策略 ⭐
├── examples/                  # 示例代码
└── docs/                      # 文档
    ├── STRATEGY_GUIDE.md     # 策略指南
    ├── ARCHITECTURE.md       # 架构文档
    └── USAGE.md              # 使用手册
```

## 🎓 Hansen 策略核心

### 目标：最大化夏普比率

```
Sharpe Ratio = 平均收益 / 收益波动率
```

### 核心原则

1. **资金保全第一** ⭐
   - 保护资本比追求收益更重要

2. **质量优于数量**
   - 宁可错过 10 个普通机会，不错过 1 个优质机会
   - 每小时 0.1-0.2 笔是正常频率

3. **严格开仓标准**
   - 信号强度 >= 80/100
   - 技术确认 >= 4/6
   - 风险回报比 >= 1:3
   - 止损强制要求

4. **自适应管理**
   - Sharpe < -0.5: 🔴 暂停交易
   - Sharpe -0.5~0: 🟡 严格控制
   - Sharpe 0~0.7: 🟢 正常运行
   - Sharpe > 0.7: 🟢 适度进取

## 📊 系统工作流

```
每 3 分钟循环:
  1. 获取账户信息和持仓
  2. 追踪绩效 → 计算夏普比率
  3. 市场分析
     ├─ 多时间框架数据 (1m/15m/1h/4h)
     ├─ 计算技术指标 (10+ 种)
     ├─ 趋势分析和评分
     └─ 入场条件验证
  4. 策略决策 (LLM + Hansen)
     ├─ 准备上下文
     ├─ Chain-of-Thought 推理
     └─ 输出结构化决策
  5. 条件执行
     └─ 只在高质量信号时交易
```

## 📚 文档

| 文档 | 说明 |
|------|------|
| [USAGE.md](USAGE.md) | 完整使用手册 |
| [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md) | Hansen 策略详解 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 架构设计文档 |
| [QUICKSTART.md](QUICKSTART.md) | 快速入门 |
| [DEEPSEEK_GUIDE.md](DEEPSEEK_GUIDE.md) | LLM 配置指南 |
| [STRATEGY_INTEGRATION.md](STRATEGY_INTEGRATION.md) | 集成完成总结 |

## 🛠️ 命令行选项

```bash
# 查看帮助
uv run python main.py --help

# 策略模式选项
uv run python main.py strategy \
  --interval 300 \              # 循环间隔（秒）
  --max-iterations 100 \        # 最大迭代次数
  --symbols BTC/USDC:USDC,ETH/USDC:USDC \  # 交易对
  --llm deepseek \               # LLM 提供商
  --debug                        # 调试模式
```

## ⚠️ 安全提示

1. **从小仓位开始** - 建议初始 < $1000
2. **监控夏普比率** - 确保 > 0 再增加仓位
3. **设置止损** - 系统强制要求
4. **定期复盘** - 每周检查交易历史

## 🔧 扩展指南

### 添加新技术指标

编辑 `src/utils/indicators.py` 和 `src/agents/strategy/market_analyzer.py`

### 调整策略

编辑 `src/prompts/hansen.txt` 修改策略逻辑

### 添加新交易所

实现 `BaseExchangeAdapter` 接口

## 💡 常见问题

**Q: 为什么一直显示 WAIT？**
A: 这是正常的！Hansen 策略强调质量，大部分时间应该是 WAIT。

**Q: 如何提高交易频率？**
A: 不建议提高频率。可以增加交易币种数量。

**Q: 系统报错怎么办？**
A: 使用 `--debug` 查看详细日志，或运行 `main.py test --debug`

## 📝 许可证

MIT