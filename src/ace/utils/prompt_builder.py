"""
Prompt Builder - 动态提示词构建器

根据知识库条目和市场数据构建 LLM Prompt
"""

from typing import Dict, List, Any
from ..models import ContextEntry, ExecutionTrace, EntryType


class PromptBuilder:
    """动态 Prompt 生成器"""

    @staticmethod
    def build_generator_prompt(
        market_data: Dict[str, Any],
        account_state: Dict[str, Any],
        entries: List[ContextEntry]
    ) -> str:
        """
        构建 Generator 的决策 Prompt

        Args:
            market_data: 市场数据（价格、指标等）
            account_state: 账户状态（余额、持仓等）
            entries: 检索到的相关知识条目

        Returns:
            完整的 Prompt 字符串
        """
        # 分类条目
        helpful_strategies = [e for e in entries if e.is_helpful() and e.entry_type == EntryType.STRATEGY]
        helpful_patterns = [e for e in entries if e.is_helpful() and e.entry_type == EntryType.PATTERN]
        risk_rules = [e for e in entries if e.entry_type == EntryType.RISK_RULE and e.confidence > 0.5]
        harmful_patterns = [e for e in entries if e.is_harmful()]

        prompt = f"""# 加密货币交易决策系统

请分析当前市场状况并做出交易决策。

## 1. 市场数据

{PromptBuilder._format_market_data(market_data)}

## 2. 账户状态

{PromptBuilder._format_account_state(account_state)}

## 3. 已验证的有效策略 ({len(helpful_strategies)} 条)

基于历史表现，以下策略已被证明有效（按置信度排序）：

{PromptBuilder._format_entries(helpful_strategies, show_confidence=True)}

## 4. 识别的市场模式 ({len(helpful_patterns)} 条)

{PromptBuilder._format_entries(helpful_patterns, show_confidence=True)}

## 5. 风控规则 ({len(risk_rules)} 条)

必须遵守的风险管理规则：

{PromptBuilder._format_entries(risk_rules, show_confidence=False)}

## 6. 需要避免的错误模式 ({len(harmful_patterns)} 条)

历史上导致亏损的操作模式，务必避免：

{PromptBuilder._format_entries(harmful_patterns, show_confidence=True, is_harmful=True)}

## 7. 决策要求

请根据以上信息，**为每个交易对**做出独立的交易决策。你的回答必须包含：

1. **市场分析**：当前市场整体状态和趋势判断
2. **决策列表**：为每个交易对给出决策和推理

### 输出格式（JSON）

```json
{{
  "analysis": "市场整体分析文本...",
  "decisions": [
    {{
      "symbol": "BTC/USDC:USDC",
      "action": "open_long|open_short|close|hold|adjust",
      "amount": 0.01,
      "leverage": 2,
      "stop_loss": 42000.0,
      "take_profit": 48000.0,
      "confidence": 0.75,
      "reasoning": "针对BTC的决策推理..."
    }},
    {{
      "symbol": "ETH/USDC:USDC",
      "action": "hold",
      "confidence": 0.6,
      "reasoning": "针对ETH的决策推理..."
    }},
    {{
      "symbol": "SOL/USDC:USDC",
      "action": "hold",
      "confidence": 0.5,
      "reasoning": "针对SOL的决策推理..."
    }}
  ]
}}
```

### 动作说明
- `open_long`: 开多仓
- `open_short`: 开空仓
- `close`: 平仓（如果有持仓）
- `hold`: 观望（无持仓）或持有（有持仓）
- `adjust`: 调整持仓（止损/止盈）

### 注意事项
- **必须为市场数据中的每个交易对都给出决策**
- 每个交易对独立分析，独立决策
- 仔细考虑账户余额和风险承受能力
- 参考已验证的有效策略
- 避免历史上失败的模式
- 严格遵守风控规则
- confidence 表示决策置信度 (0-1)
- reasoning 必须解释该币对的具体分析
"""

        return prompt

    @staticmethod
    def build_reflector_prompt(trace: ExecutionTrace) -> str:
        """
        构建 Reflector 的反思 Prompt

        Args:
            trace: 执行轨迹

        Returns:
            反思 Prompt
        """
        decisions = trace.decisions
        success = "成功" if trace.execution_success else "失败"
        pnl_str = ""

        if trace.pnl:
            pnl_sign = "+" if float(trace.pnl) > 0 else ""
            pnl_str = f"\n- 盈亏: {pnl_sign}${trace.pnl}"

        # 格式化所有决策
        decisions_str = ""
        for i, dec in enumerate(decisions, 1):
            decisions_str += f"\n{i}. {dec.symbol}:\n"
            decisions_str += f"   - 动作: {dec.action}\n"
            decisions_str += f"   - 置信度: {dec.confidence:.2f}\n"
            decisions_str += f"   - 推理: {dec.reasoning}\n"

        prompt = f"""# 交易反思分析

请对以下交易过程进行深度反思和分析。

## 1. 市场条件

{PromptBuilder._format_market_data(trace.market_data)}

## 2. 决策过程

### 使用的策略
本次决策参考了 {len(trace.retrieved_entries)} 个知识条目。

### 决策内容（共 {len(decisions)} 个交易对）
{decisions_str}

## 3. 执行结果

- 状态: {success}
- 执行信息: {trace.execution_results}{pnl_str}
- 错误: {'; '.join(trace.execution_errors) if trace.execution_errors else '无'}

## 4. 账户变化

{PromptBuilder._format_account_change(trace.account_change)}

## 5. 反思任务

请从以下角度进行分析：

### A. 策略有效性评估
对本次使用的 {len(trace.retrieved_entries)} 个策略条目，逐一评估：
- 哪些策略是有用的（helpful）？为什么？
- 哪些策略是有害的（harmful）？为什么？
- 哪些策略影响不明显（neutral）？

### B. 失败诊断（如果失败）
如果交易失败或亏损，请诊断失败类型：
- **conceptual**: 对市场理解错误
- **computational**: 计算或指标错误
- **strategic**: 策略选择或时机错误
- **execution**: 执行层面的技术问题

### C. 洞察提取
发现了哪些新的规律或模式？

### D. 错误模式识别
有哪些操作应该避免？

### 输出格式（JSON）

```json
{{
  "is_successful": true/false,
  "failure_type": "conceptual|computational|strategic|execution|none",
  "strategy_evaluations": [
    {{"entry_id": "entry_xxx", "is_helpful": true, "reason": "..."}}
  ],
  "key_insights": [
    "新发现的规律1",
    "新发现的规律2"
  ],
  "error_patterns": [
    "应该避免的错误1",
    "应该避免的错误2"
  ],
  "reflection_summary": "总体反思总结..."
}}
```
"""

        return prompt

    @staticmethod
    def _format_market_data(data: Dict[str, Any]) -> str:
        """格式化市场数据"""
        if not data:
            return "（暂无市场数据）"

        lines = []
        for symbol, info in data.items():
            lines.append(f"### {symbol}")

            if isinstance(info, dict):
                if 'price' in info:
                    lines.append(f"- 当前价格: ${info['price']:.2f}")
                if 'change_24h' in info:
                    lines.append(f"- 24h 涨跌: {info['change_24h']:+.2f}%")
                if 'volume' in info:
                    lines.append(f"- 成交量: ${info['volume']:,.0f}")
                if 'indicators' in info:
                    lines.append(f"- 技术指标:")
                    for k, v in info['indicators'].items():
                        lines.append(f"  - {k}: {v}")

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_account_state(state: Dict[str, Any]) -> str:
        """格式化账户状态"""
        if not state:
            return "（暂无账户信息）"

        lines = []

        if 'balance' in state:
            balance = state['balance']
            lines.append(f"- 总资产: ${balance.get('total', 0):.2f}")
            lines.append(f"- 可用资金: ${balance.get('available', 0):.2f}")
            lines.append(f"- 冻结保证金: ${balance.get('frozen', 0):.2f}")

        if 'positions' in state:
            positions = state['positions']
            if positions:
                lines.append(f"\n当前持仓 ({len(positions)} 个):")
                for pos in positions:
                    side = "多" if pos.get('side') == 'long' else "空"
                    pnl = pos.get('unrealized_pnl', 0)
                    lines.append(
                        f"  - {pos.get('symbol')}: {side}仓, "
                        f"数量 {pos.get('amount')}, "
                        f"未实现盈亏 ${pnl:+.2f}"
                    )
            else:
                lines.append("\n当前无持仓")

        if 'statistics' in state:
            stats = state['statistics']
            if stats:
                lines.append(f"\n交易统计:")
                lines.append(f"- 累计平仓: {stats.get('total_positions', 0)} 次")
                lines.append(f"- 胜率: {stats.get('win_rate', 0) * 100:.1f}%")
                lines.append(f"- 总盈亏: ${stats.get('total_pnl', 0):+.2f}")

        return "\n".join(lines)

    @staticmethod
    def _format_account_change(change: Dict[str, Any]) -> str:
        """格式化账户变化"""
        if not change:
            return "（无变化）"

        lines = []
        if 'pnl' in change:
            lines.append(f"- 本次盈亏: ${change['pnl']:+.2f}")
        if 'balance_before' in change and 'balance_after' in change:
            lines.append(f"- 账户余额: ${change['balance_before']:.2f} → ${change['balance_after']:.2f}")

        return "\n".join(lines) if lines else "（无明显变化）"

    @staticmethod
    def _format_entries(
        entries: List[ContextEntry],
        show_confidence: bool = True,
        is_harmful: bool = False
    ) -> str:
        """格式化知识条目"""
        if not entries:
            return "（无相关条目）"

        # 按置信度排序
        sorted_entries = sorted(entries, key=lambda e: e.confidence, reverse=not is_harmful)

        lines = []
        for i, entry in enumerate(sorted_entries[:10], 1):  # 最多显示 10 条
            conf_str = f" (置信度: {entry.confidence:.2f})" if show_confidence else ""
            lines.append(f"{i}. [{entry.entry_type.value}]{conf_str}")
            lines.append(f"   {entry.content}")
            lines.append("")

        return "\n".join(lines)
