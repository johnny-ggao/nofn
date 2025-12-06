# NOFN 量化交易系统

基于 LLM 的智能量化交易系统，采用模块化架构设计。

## 核心特性

- **LLM 驱动决策**: 使用大语言模型进行交易决策，支持多种 LLM 提供商
- **策略模板**: 内置多种交易策略模板（稳健、激进、资金费率套利等）
- **多时间框架分析**: 支持 MTF 技术指标（1H/15M/5M 多周期）
- **反思模式**: 自动分析历史表现，动态调整策略参数
- **实盘/模拟**: 支持真实交易和虚拟交易模式

## 项目结构

```
nofn/
├── main.py                 # 主程序入口
├── config/
│   └── config.yaml         # 配置文件
├── src/
│   ├── config.py           # 配置管理
│   ├── strategy/           # 策略层
│   │   ├── agent.py        # 策略代理 (StrategyAgent)
│   │   └── __init__.py
│   └── trading/            # 交易核心
│       ├── models.py       # 数据模型
│       ├── engine.py       # 交易引擎 (StrategyRuntime, DecisionCoordinator)
│       ├── market/         # 市场数据与特征
│       │   ├── data_source.py      # CCXT 数据源
│       │   ├── candle.py           # K线特征计算
│       │   ├── mtf_candle.py       # 多时间框架指标
│       │   ├── market_snapshot.py  # 行情快照
│       │   └── pipeline.py         # 特征管道
│       ├── decision/       # 决策模块
│       │   ├── llm_composer.py     # LLM 决策器
│       │   └── system_prompt.py    # 系统提示词
│       ├── execution/      # 交易执行
│       │   ├── ccxt_trading.py     # CCXT 实盘执行
│       │   └── paper_trading.py    # 模拟交易
│       ├── portfolio/      # 仓位管理
│       ├── history/        # 历史记录与摘要
│       ├── reflection/     # 反思模式
│       │   ├── analyzer.py         # 表现分析器
│       │   └── composer.py         # 反思决策器
│       └── templates/      # 策略模板
│           ├── default.txt
│           ├── aggressive.txt
│           ├── insane.txt
│           └── funding_rate.txt
└── tests/
    └── test_new_architecture.py
```

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置

复制并编辑配置文件：

```bash
cp config/config.yaml.example config/config.yaml
```

创建 `.env` 文件配置敏感信息：

```bash
# 交易所 API
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key

# LLM API
OPENROUTER_API_KEY=your_api_key
# 或
OPENAI_API_KEY=your_api_key
```

### 3. 运行

```bash
# 查看可用模板
python main.py --list-templates

# 使用默认配置运行
python main.py

# 指定模板和交易对
python main.py -t aggressive -s BTC/USDT:USDT ETH/USDT:USDT

# 启用反思模式
python main.py --reflection
```

## 命令行参数

| 参数 | 说明 |
|------|------|
| `-t, --template` | 策略模板: default, aggressive, insane, funding_rate |
| `-s, --symbols` | 交易对列表，如 `BTC/USDT:USDT ETH/USDT:USDT` |
| `-m, --mode` | 交易模式: live (实盘) 或 virtual (模拟) |
| `-i, --interval` | 决策间隔（秒） |
| `-r, --reflection` | 启用反思模式 |
| `--list-templates` | 列出可用模板 |

## 架构设计

### 核心流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Market     │ -> │  Decision   │ -> │  Execution  │
│  (数据获取)  │    │  (LLM决策)   │    │  (交易执行)  │
└─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │
       v                  v                  v
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Features   │    │  Templates  │    │  Portfolio  │
│  (特征计算)  │    │  (策略模板)  │    │  (仓位管理)  │
└─────────────┘    └─────────────┘    └─────────────┘
```

### 模块说明

| 模块 | 职责 |
|------|------|
| `strategy/` | 策略代理，编排交易循环 |
| `trading/engine.py` | 交易引擎，协调各组件 |
| `trading/market/` | 市场数据获取与特征计算 |
| `trading/decision/` | LLM 决策器 |
| `trading/execution/` | 订单执行（实盘/模拟） |
| `trading/portfolio/` | 仓位与资金管理 |
| `trading/history/` | 交易历史与表现摘要 |
| `trading/reflection/` | 反思分析与策略调整 |

## 策略模板

### default (稳健顺势)

- 顺势交易，只在明确趋势中开仓
- 置信度要求 > 0.7
- 适合中长线持仓

### aggressive (激进动量)

- 追逐短期动量
- 置信度要求 > 0.6
- 高频交易风格

### insane (极端激进)

- 最大化交易频率
- 置信度要求 > 0.5
- 高风险高收益

### funding_rate (资金费率套利)

- 专注资金费率套利
- 在费率极端时开仓
- 相对低风险策略

## 反思模式

启用 `--reflection` 后，系统会：

1. **分析历史表现**: 计算夏普比、胜率、最大回撤等指标
2. **识别问题模式**: 检测过度交易、连续亏损、单一标的集中等
3. **动态调整**:
   - 自动调整置信度阈值
   - 过滤表现差的标的
   - 严重问题时触发冷静期

## 技术指标

### 多时间框架 (MTF) 指标

**1H 周期:**
- EMA(7, 21, 55)
- MACD(6, 13, 5)
- RSI(14), ADX(14), ATR(14)
- Bollinger Bands(20, 2)

**15M 周期:**
- EMA(8, 21, 50)
- MACD(6, 13, 5)
- RSI(14), Stochastic(14, 3, 3)
- Volume ROC

**5M 周期:**
- EMA(8), RSI(9)
- MACD(5, 10, 3)
- Volume MA

## 配置示例

```yaml
# config/config.yaml
exchange:
  id: binance
  market_type: swap      # spot, future, swap
  margin_mode: cross     # cross, isolated
  testnet: false

strategy:
  name: "NOFN Strategy"
  template: default
  symbols:
    - BTC/USDT:USDT
    - ETH/USDT:USDT
  trading_mode: virtual  # live, virtual
  initial_capital: 10000
  max_leverage: 3
  max_positions: 5
  decide_interval: 60

llm:
  provider: openrouter
  model: deepseek/deepseek-chat
  temperature: 0.3
```

## 开发

### 运行测试

```bash
uv run pytest tests/ -v
```

### 添加新策略模板

1. 在 `src/trading/templates/` 创建新的 `.txt` 文件
2. 按照现有模板格式编写策略提示词
3. 使用 `-t your_template` 运行

### 自定义特征计算

继承 `CandleBasedFeatureComputer` 实现自定义指标：

```python
from src.trading.market import CandleBasedFeatureComputer

class MyFeatureComputer(CandleBasedFeatureComputer):
    def compute_features(self, candles, meta=None):
        # 计算自定义指标
        ...
```

## 许可证

MIT
