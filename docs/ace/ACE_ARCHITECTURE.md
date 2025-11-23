# ACE Trading Agent Architecture

## 概述

基于 Agentic Context Engineering (ACE) 框架构建的自进化交易智能体系统。

## 核心理念

将上下文从"静态提示词"转变为"动态演化的结构化知识库"，通过 Generator-Reflector-Curator 三角循环实现智能体的持续自我优化。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     ACE Trading Agent                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │  Generator   │─────>│  Reflector   │─────>│  Curator  │ │
│  │              │      │              │      │           │ │
│  │ • 市场分析    │      │ • 结果分析    │      │ • 提取策略 │ │
│  │ • 决策推理    │      │ • 失败诊断    │      │ • 更新知识 │ │
│  │ • 执行交易    │      │ • 标记模式    │      │ • 去重合并 │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                                            │       │
│         │                                            │       │
│         v                                            v       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Context Store (知识库)                   │  │
│  │                                                       │  │
│  │  • Playbook Entries (策略条目)                        │  │
│  │  • Market Patterns (市场模式)                         │  │
│  │  • Risk Rules (风控规则)                              │  │
│  │  • Error Patterns (错误模式)                          │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                    │
│         v                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Dynamic Prompt Builder (动态提示词构建)        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
         │                              │
         v                              v
┌─────────────────┐          ┌──────────────────┐
│  Trading Engine │          │  Market Data API │
│  (执行层)        │          │  (数据源)         │
└─────────────────┘          └──────────────────┘
```

## 核心组件

### 1. Context Entry (知识条目)

每个条目是知识库中的原子单元：

```python
@dataclass
class ContextEntry:
    entry_id: str           # 唯一标识
    entry_type: str         # 类型：strategy/pattern/rule/error
    content: str            # 条目内容

    # ACE 元数据
    helpful_count: int = 0  # 有用次数
    harmful_count: int = 0  # 有害次数
    confidence: float = 0.5 # 置信度

    # 时间信息
    created_at: datetime
    last_used: Optional[datetime] = None

    # 关联信息
    source_case_ids: List[str] = []  # 来源案例
    tags: List[str] = []             # 标签
```

### 2. Generator (生成者)

**职责**：
- 接收市场数据和当前 Playbook
- 使用 LLM 进行市场分析和决策推理
- 执行交易操作
- 记录完整的推理轨迹（reasoning trace）

**输入**：
- 市场快照（价格、指标、K线等）
- 当前 Playbook（相关策略条目）
- 账户状态

**输出**：
- 交易决策
- 推理过程
- 执行结果

### 3. Reflector (反思者)

**职责**：
- 分析 Generator 的执行结果
- 诊断失败原因（conceptual/computational/strategic）
- 标记哪些策略有效/无效
- 提取新的洞察

**输入**：
- Generator 的推理轨迹
- 执行结果（成功/失败）
- 账户变化（盈亏、持仓等）

**输出**：
- Reflection 报告
- 策略有效性评分
- 新发现的模式

### 4. Curator (策展者)

**职责**：
- 将 Reflector 的洞察转化为结构化条目
- 增量式更新 Playbook
- 去重和合并相似条目
- 管理知识库生命周期

**核心操作**：
- `add_entry()`: 添加新条目
- `update_entry()`: 更新现有条目的元数据
- `merge_entries()`: 合并语义相似的条目
- `prune_entries()`: 清理低置信度条目

### 5. Context Store (知识库)

**存储结构**：
```sql
-- playbook_entries 表
CREATE TABLE playbook_entries (
    entry_id TEXT PRIMARY KEY,
    entry_type TEXT,
    content TEXT,
    helpful_count INTEGER,
    harmful_count INTEGER,
    confidence REAL,
    created_at TEXT,
    last_used TEXT,
    tags TEXT,  -- JSON array
    embedding BLOB  -- 向量表示
);

-- execution_traces 表
CREATE TABLE execution_traces (
    trace_id TEXT PRIMARY KEY,
    timestamp TEXT,
    market_data TEXT,  -- JSON
    decision TEXT,
    execution_result TEXT,  -- JSON
    reflection TEXT,
    related_entries TEXT  -- JSON array of entry_ids
);
```

## 工作流程

### 主循环（每次迭代）

```python
1. Generator Phase:
   a. 获取市场数据
   b. 检索相关 Playbook Entries (基于相似度)
   c. 构建动态 Prompt
   d. LLM 推理并决策
   e. 执行交易
   f. 记录执行轨迹

2. Reflector Phase:
   a. 获取执行结果和账户变化
   b. 分析决策质量
   c. 标记使用的策略是否有效
   d. 诊断失败原因（如有）
   e. 提取新洞察

3. Curator Phase:
   a. 接收 Reflection 结果
   b. 创建新的 Context Entries（如有新模式）
   c. 更新现有 Entries 的元数据
      - 有效策略: helpful_count += 1
      - 无效策略: harmful_count += 1
   d. 触发去重和压缩（周期性）

4. Sleep / Wait for next iteration
```

## 关键特性

### 1. 增量式更新（Delta Updates）

不重写整个 prompt，而是：
- 添加新条目
- 更新计数器
- 局部编辑

### 2. 语义去重

使用 Embeddings 检测相似条目：
```python
if cosine_similarity(new_entry, existing_entry) > 0.9:
    merge_entries(new_entry, existing_entry)
```

### 3. 动态 Prompt 生成

每次决策时构建：
```
Base Template
+
Helpful Entries (confidence > 0.7)
+
Harmful Patterns to Avoid (harmful_count > helpful_count)
+
Recent Market Context
```

### 4. 自适应清理

当知识库过大时：
- 删除 confidence < 0.3 的条目
- 合并相似条目
- 归档长期未使用的条目

## 数据流

```
Market Data → Generator → Decision
                    ↓
            Execution Result
                    ↓
               Reflector → Insights
                    ↓
                Curator → Updated Playbook
                    ↓
                Context Store
                    ↓
         (Next iteration) Generator
```

## 与原系统对比

| 原系统 | ACE 系统 |
|--------|----------|
| 固定 nofn_v2.txt | 动态 Playbook |
| 案例整体存储 | 结构化条目 + 元数据 |
| 简单检索 | 有效性评分 + 相似度 |
| 被动记忆 | 主动筛选和优化 |
| LangGraph | 纯 ACE 工作流 |

## 技术栈

- **LLM**: OpenAI/Anthropic API
- **向量存储**: SQLite + numpy (轻量级)
- **Embeddings**: OpenAI text-embedding-3-small
- **交易执行**: 复用现有 adapters (HyperliquidAdapter)
- **数据库**: SQLite (ace.db)

## 预期效果

基于 ACE 论文的结果：
- 交易决策质量提升 8-10%
- 适应新市场速度提升 87%
- 减少重复性错误
- 自动积累可复用策略
