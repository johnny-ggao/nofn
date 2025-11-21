# NoFn - LangGraph 驱动的自主交易系统

基于 LangGraph 构建的智能交易系统，实现交易执行、决策制定和持续学习的三层分离架构。

## ✨ 核心特性

### 🏗️ 三层分离架构

**Layer 1: 执行层 (TradingEngine)**
- 快速、确定性的交易执行
- 批量获取市场数据和技术指标
- 毫秒级响应，无 LLM 参与

**Layer 2: 决策层 (DecisionMaker)**
- LLM 驱动的交易决策
- 基于市场数据和历史记忆
- 结构化输出（JSON 格式交易信号）

**Layer 3: 学习层 (LearningGraph)**
- LangGraph 编排的学习工作流
- SqliteSaver 持久化 checkpointer（状态管理、断点恢复）
- MemoryManager 记忆检索（SQLite 相似度匹配）
- 反思学习、持续进化和策略优化

### 🔄 LangGraph 工作流

```
Market Data → Retrieve Memory → Decide → Execute → Reflect → Update Memory
```

每个节点专注于特定职责：
- `get_market_data`: 获取市场快照
- `retrieve_memory`: 检索相似历史案例
- `decide`: LLM 分析并生成交易信号
- `execute`: 执行交易操作
- `reflect`: 反思本次决策
- `update_memory`: 更新记忆库

### 📊 完整技术分析

- **批量数据获取** - 一次 API 调用获取所有数据
- **10+ 技术指标** - EMA、MACD、RSI、ATR、OBV、Stochastic
- **趋势识别** - 多头/空头/中性排列
- **智能持仓管理** - 动态阈值自动更新止损止盈

### 🧠 分层记忆与学习

- **三层记忆架构** - 短期案例、周期摘要、长期经验库
- **智能压缩** - LLM 自动生成记忆摘要，压缩率 99.7%
- **自动清理** - 保留有价值案例，归档历史数据
- **相似度检索** - 基于市场特征的相似度匹配（SQLite 存储）
- **状态持久化** - SqliteSaver checkpointer 支持断点恢复
- **反思机制** - LLM 分析决策质量并提取经验
- **经验积累** - 持续优化交易策略

**存储效率**：
- 短期（7天）：详细案例
- 中期（每周）：LLM 生成的总结和模式
- 长期：核心经验库
- 年度存储：~3MB（vs 原始 982MB）

### 🤖 LLM 增强决策

- **多 LLM 支持** - DeepSeek、OpenAI、Anthropic
- **上下文感知** - 结合市场数据和历史经验
- **结构化输出** - JSON 格式交易信号
- **自然语言解释** - 每个决策都有清晰理由

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

复制配置模板并填写你的 API 密钥：

```bash
cp config/config.yaml.example config/config.yaml
```

编辑 `config/config.yaml`：

```yaml
# LLM 配置
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  api_key: "your_api_key_here"
  base_url: "https://api.deepseek.com"
  temperature: 0.7

# 交易所配置
exchanges:
  hyperliquid:
    api_key: "your_wallet_address_here"
    api_secret: "your_private_key_here"
    testnet: false

# 策略配置
strategy:
  exchange: "hyperliquid"
  symbols:
    - "BTC/USDC:USDC"
    - "ETH/USDC:USDC"
    - "SOL/USDC:USDC"
  interval_seconds: 180
  max_iterations: null  # null = 无限循环

# 风险管理
risk:
  max_position_size: 10000.0
  max_leverage: 10
  max_daily_trades: 50
  enable_risk_check: true
```

### 3. 运行

```bash
# 启动交易系统
uv run python main.py

# 运行指定次数后停止
uv run python main.py --max-iterations 10

# 查看帮助
uv run python main.py --help

# 查看交易记录和持仓
uv run python view_trades.py --type all        # 查看所有信息
uv run python view_trades.py --type open       # 查看当前持仓
uv run python view_trades.py --type closed     # 查看已平仓记录
uv run python view_trades.py --type stats      # 查看交易统计
uv run python view_trades.py --days 7 --limit 20  # 查看最近7天的20条记录
```

## 📖 架构详解

### 三层架构设计原则

**设计理念：将 LangGraph 用在真正重要的地方**

- ❌ **不用于**：简单的工具调用、确定性操作
- ✅ **用于**：状态管理、记忆检索、反思学习、策略优化

### Layer 1: 执行层 (TradingEngine)

**职责**：快速、确定性的数据获取和交易执行

```python
# 批量获取市场快照
snapshot = await engine.get_market_snapshot(symbols)

# 包含：
# - 实时价格
# - 持仓信息
# - 技术指标（EMA、RSI、MACD等）
# - 账户状态
```

**特点**：
- 毫秒级响应
- 批量 API 调用
- 无 LLM 推理
- 纯 Python/NumPy 计算

### Layer 2: 决策层 (DecisionMaker)

**职责**：基于市场数据和记忆进行交易决策

```python
decision = await decision_maker.analyze_and_decide(
    snapshot,
    memory_context
)

# 输出结构化决策：
# {
#   "decision_type": "trade" | "hold" | "close",
#   "signals": [
#     {
#       "action": "open_long",
#       "symbol": "BTC/USDC:USDC",
#       "amount": 0.001,
#       "leverage": 3,
#       "stop_loss": 88000.0,
#       "take_profit": 96000.0,
#       "confidence": 85,
#       "reason": "..."
#     }
#   ]
# }
```

**特点**：
- LLM 驱动
- 结合历史经验
- 结构化输出
- 清晰推理过程

### Layer 3: 学习层 (LearningGraph)

**职责**：使用 LangGraph 编排学习和进化流程

```python
# LangGraph 工作流
workflow = StateGraph(TradingState)
workflow.add_node("get_market_data", self._get_market_data)
workflow.add_node("retrieve_memory", self._retrieve_memory)
workflow.add_node("decide", self._decide)
workflow.add_node("execute", self._execute)
workflow.add_node("reflect", self._reflect)
workflow.add_node("update_memory", self._update_memory)

# 顺序执行
workflow.set_entry_point("get_market_data")
workflow.add_edge("get_market_data", "retrieve_memory")
workflow.add_edge("retrieve_memory", "decide")
workflow.add_edge("decide", "execute")
workflow.add_edge("execute", "reflect")
workflow.add_edge("reflect", "update_memory")
workflow.add_edge("update_memory", END)

# 使用 SqliteSaver 作为 Checkpointer（持久化状态）
checkpointer = SqliteSaver.from_conn_string(db_path)
graph = workflow.compile(checkpointer=checkpointer)
```

**特点**：
- **SqliteSaver Checkpointer**: 持久化图状态，支持断点恢复
- **MemoryManager**: SQLite 记忆存储，相似度匹配历史案例
- **统一数据库**: 所有组件共享 `data/nofn.db`
- LLM 反思总结
- 经验提取和存储

### 分层记忆系统

**三层架构**：

1. **短期记忆（7天）**
   - 存储详细的交易案例
   - 包含完整市场条件、决策、执行结果
   - 用于实时决策参考

2. **中期记忆（每周摘要）**
   - LLM 自动生成周度总结
   - 提取关键模式、成功策略、失败教训
   - 压缩 ~50 个案例 → 1 个摘要

3. **长期记忆（核心经验库）**
   - 历史摘要归档
   - 持久化核心交易模式
   - 跨周期策略优化

**存储结构 - 详细案例**：
```json
{
  "case_id": "case_1234567890",
  "market_conditions": {
    "assets": {...},
    "account": {...}
  },
  "decision": "LLM 的完整分析...",
  "execution_result": [...],
  "reflection": "反思内容...",
  "lessons_learned": ["经验1", "经验2"],
  "timestamp": "2025-11-20T..."
}
```

**存储结构 - 周度摘要**：
```json
{
  "summary_id": "weekly_20251120",
  "period": "2025-11-13 ~ 2025-11-20",
  "statistics": {
    "total_trades": 15,
    "win_rate": 0.67,
    "sharpe_ratio": 1.8
  },
  "key_patterns": [
    "BTC在90k支撑位反弹概率高",
    "夜间波动率通常较低"
  ],
  "successful_strategies": [
    "趋势跟随 + 动态止盈"
  ],
  "lessons": [
    "震荡市场减少开仓频率",
    "止损不宜设置过紧"
  ],
  "market_insights": "本周市场呈现区间震荡..."
}
```

**数据库存储（Phase 1 架构）**：
- 使用 SQLite 统一数据库（`data/nofn.db`）
- 表结构：
  - `trades`: 交易记录（TradeHistoryManager）
  - `positions`: 持仓记录（TradeHistoryManager）
  - `trading_cases`: 交易案例（MemoryManager）
  - `memory_summaries`: 记忆摘要（MemoryManager）
  - `checkpoints`: LangGraph 状态快照（SqliteSaver）
- 线程安全的连接管理
- SQL 索引优化查询性能

**检索与压缩**：
- 基于市场条件的相似度匹配
- 智能清理：保留最近 1000 个案例（30天内）
- 自动生成周度/月度摘要
- 压缩效率：99.7%（982MB → 3MB/年）

## 📁 项目结构

```
nofn/
├── main.py                    # 系统入口点 ⭐
├── generate_summary.py        # 记忆摘要生成工具
├── config/
│   ├── config.yaml.example    # 配置模板
│   └── config.yaml           # 实际配置（不提交）
├── src/
│   ├── engine/
│   │   ├── trading_engine.py         # Layer 1: 执行层 ⭐
│   │   └── market_snapshot.py        # 市场快照模型
│   ├── decision/
│   │   └── decision_maker.py         # Layer 2: 决策层 ⭐
│   ├── learning/
│   │   ├── learning_graph.py         # Layer 3: 学习层 ⭐
│   │   └── memory_manager.py         # 记忆管理
│   ├── adapters/
│   │   ├── base.py                   # 交易所适配器基类
│   │   └── hyperliquid.py            # Hyperliquid 实现
│   ├── models/
│   │   └── trading.py                # 交易模型定义
│   └── utils/
│       ├── indicators.py             # 技术指标计算 ⭐
│       └── config.py                 # 配置管理
└── data/
    └── nofn.db                       # 统一 SQLite 数据库 ⭐
        ├── trades                    # 交易记录表 (TradeHistoryManager)
        ├── positions                 # 持仓记录表 (TradeHistoryManager)
        ├── trading_cases             # 交易案例表 (MemoryManager)
        ├── memory_summaries          # 记忆摘要表 (MemoryManager)
        └── checkpoints               # LangGraph 状态快照 (SqliteSaver)
```

## 🔄 系统工作流

### 单次迭代流程

```
1. 获取市场数据 (Layer 1)
   └─ 批量获取价格、持仓、技术指标

2. 检索历史记忆 (Layer 3)
   └─ 查找相似市场条件的案例

3. 决策分析 (Layer 2)
   ├─ LLM 分析市场 + 历史经验
   └─ 输出结构化交易信号

4. 执行交易 (Layer 1)
   ├─ 开仓/平仓
   ├─ 设置止损止盈
   └─ 更新持仓 SL/TP

5. 反思总结 (Layer 3)
   ├─ LLM 评估决策质量
   └─ 提取经验教训

6. 更新记忆 (Layer 3)
   └─ 保存完整交易案例
```

### 持续学习循环

```
每 3 分钟:
  运行一次完整迭代
  └─ 积累经验
  └─ 优化策略
  └─ 提升决策质量
```

## 🎯 交易信号类型

### 开仓信号
- `open_long`: 开多头仓位
- `open_short`: 开空头仓位

### 持仓管理
- `hold`: 持有现有仓位（可更新止损止盈）
- `close_position`: 平仓
- `set_stop_loss_take_profit`: 修改止损止盈

### 观望
- `wait`: 无交易机会，等待

## 🛡️ 风险管理

### 强制风险控制

1. **止损止盈强制**
   - 所有开仓信号必须包含止损
   - 推荐风险回报比 >= 1:2

2. **智能持仓监控**
   - 自动检测缺失的止损止盈
   - 动态阈值防止频繁微调
   - 只在价格显著变化时更新（>0.3%）

3. **杠杆限制**
   - 配置最大杠杆（默认 10x）
   - 基于波动率动态调整

4. **仓位限制**
   - 最大持仓金额控制
   - 每日交易次数限制（仅计算真实交易）

### 智能止损止盈更新

系统使用**动态阈值机制**避免频繁调整：

```python
# 智能阈值计算
current_price = 90000.0  # BTC 当前价格
min_threshold = max(1.0, current_price * 0.003)  # 270 美元

# 只在显著变化时更新
if signal.action == 'hold':
    if stop_loss_missing:
        # 缺失止损 → 立即添加
        update_stop_loss()
    elif abs(new_sl - current_sl) > min_threshold:
        # 差异 > 270 美元 → 更新
        update_stop_loss()
    else:
        # 微小差异 → 保持稳定
        keep_current_stop_loss()
```

**效果**：
- 避免 LLM 微小差异导致的频繁更新
- 减少不必要的 API 调用
- 提升系统稳定性
- 降低交易成本

## 🔧 技术指标

系统使用的技术指标：

- **EMA** (20/50/200): 趋势识别
- **RSI** (14): 超买超卖
- **MACD**: 动量和趋势
- **ATR** (14): 波动率
- **OBV**: 成交量趋势
- **Stochastic**: 震荡指标

所有指标都是批量计算，使用 NumPy 优化性能。

## 📊 性能监控

### 账户统计

系统实时追踪（**仅计算真实交易**）：
- 夏普比率（Sharpe Ratio）
- 总收益率
- 胜率
- 平均盈亏
- 最大回撤
- **交易频率**（不含止损止盈修改）

**注意**：`hold` 和 `set_stop_loss_take_profit` 动作不计入交易统计，避免夸大交易频率。

### 学习进度

- 已保存案例数（自动清理 > 1000）
- 历史摘要数量
- 经验教训数量
- 决策质量趋势

## ⚠️ 安全提示

1. **配置文件安全**
   - ⚠️ `config/config.yaml` 包含私钥，已在 `.gitignore` 中
   - 永远不要提交包含真实密钥的配置文件
   - 使用 `config.yaml.example` 作为模板

2. **从小仓位开始**
   - 建议初始资金 < $500 进行测试
   - 观察系统运行 1-2 周

3. **监控系统行为**
   - 查看日志文件 `logs/trading.log`
   - 定期检查 `data/memory/cases.json`

4. **测试网先行**
   - 在配置中设置 `testnet: true`
   - 测试稳定后再使用主网

## 🔧 配置说明

### LLM 配置

支持的提供商：
- **DeepSeek** (推荐): 性价比高，中文支持好
- **OpenAI**: GPT-4 系列
- **Anthropic**: Claude 系列

```yaml
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  api_key: "sk-..."
  temperature: 0.7  # 0.0-2.0，越高越随机
```

### 策略配置

```yaml
strategy:
  exchange: "hyperliquid"
  symbols:
    - "BTC/USDC:USDC"   # 添加更多交易对
    - "ETH/USDC:USDC"
  interval_seconds: 180  # 3分钟一次迭代
  max_iterations: null   # null = 无限运行
```

### 记忆管理配置

```yaml
memory:
  max_cases: 1000           # 最大保留案例数
  keep_recent_days: 30      # 保留最近N天的所有案例
  enable_auto_cleanup: true # 启用自动清理
  enable_archiving: true    # 启用归档功能
```

### 风险配置

```yaml
risk:
  max_position_size: 10000.0  # 单个仓位最大金额
  max_leverage: 10            # 最大杠杆
  max_daily_trades: 50        # 每日最大交易次数
  enable_risk_check: true     # 启用风控检查
```

## 🚀 命令行选项

### 运行交易系统

```bash
# 基础运行
uv run python main.py

# 运行指定次数
uv run python main.py --max-iterations 10

# 自定义配置文件
uv run python main.py --config path/to/config.yaml

# 查看帮助
uv run python main.py --help
```

### 生成记忆摘要

```bash
# 手动生成每周摘要
uv run python generate_summary.py

# 查看摘要详情
cat data/memory/summaries.json | jq .
```

## 💡 常见问题

**Q: 为什么一直显示 WAIT？**
A: 这是正常的！系统会等待高质量的交易机会。可以：
- 增加监控的交易对数量
- 降低决策层的入场门槛（不推荐）

**Q: 如何查看系统决策过程？**
A: 系统会打印完整的分析过程，也可以查看：
- 控制台输出（彩色日志）
- `data/memory/cases.json` 文件

**Q: 系统会自动止损吗？**
A: 是的！系统会：
- 在开仓时自动设置止损止盈
- 在 hold 信号中更新缺失的 SL/TP
- 交易所会自动触发止损单

**Q: 可以同时监控多个交易所吗？**
A: 目前只支持 Hyperliquid，但架构设计支持扩展。实现 `BaseExchangeAdapter` 接口即可添加新交易所。

**Q: 记忆会占用多少空间？**
A: 系统采用分层记忆架构：
- 短期（7天）：约 5-10MB
- 中期摘要：约 100-200KB
- 长期归档：自动归档，按月分组
- **总存储（1年）**：约 3MB（vs 未压缩的 982MB）

系统会**自动清理和归档**：
- 保留最近 30 天的所有案例
- 旧案例只保留有交易执行的
- 被清理的案例归档到 `data/memory/archives/`
- 每周自动生成 LLM 摘要

**Q: 如何生成记忆摘要？**
A: 系统会自动在需要时生成摘要，也可以手动执行：
```bash
uv run python generate_summary.py
```

## 📚 扩展开发

### 添加新交易所

实现 `BaseExchangeAdapter` 接口：

```python
class NewExchangeAdapter(BaseExchangeAdapter):
    async def open_position(self, ...): ...
    async def close_position(self, ...): ...
    async def get_position(self, ...): ...
    async def modify_stop_loss_take_profit(self, ...): ...
```

### 自定义决策策略

修改 `src/decision/decision_maker.py` 中的提示词模板。

### 添加新的学习节点

在 `src/learning/learning_graph.py` 中添加新节点：

```python
workflow.add_node("your_node", self._your_method)
workflow.add_edge("decide", "your_node")
workflow.add_edge("your_node", "execute")
```

## 📝 许可证

MIT

## 🙏 致谢

- **LangGraph** - 强大的状态管理和工作流编排
- **CCXT** - 统一的交易所 API
- **DeepSeek** - 高性价比的 LLM 服务
