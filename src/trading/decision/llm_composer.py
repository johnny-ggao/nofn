"""LLM-driven composer using LangGraph (replacing Agno).

This module provides the LlmComposer that uses LangChain/LangGraph for
structured LLM outputs instead of Agno.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from termcolor import cprint
from pydantic import BaseModel, Field, ValidationError

from ..models import (
    ComposeContext,
    ComposeResult,
    PriceMode,
    TradeDecisionAction,
    TradeDecisionItem,
    TradeInstruction,
    TradePlanProposal,
    UserRequest,
    derive_side_from_action,
    get_current_timestamp_ms,
)
from .interfaces import BaseComposer
from .system_prompt import SYSTEM_PROMPT
from .llm_factory import create_llm_from_config

# Import template loader
from ..templates import (
    TemplateNotFoundError,
    get_template_loader,
)


class SingleSymbolDecision(BaseModel):
    """单个交易对的 LLM 决策输出。"""

    action: str = Field(
        ...,
        description="操作类型: open_long|open_short|close_long|close_short|noop",
    )
    target_qty: float = Field(
        default=0.0,
        description="操作数量（正数）",
    )
    leverage: float = Field(
        default=3.0,
        description="杠杆倍数",
    )
    sl_price: Optional[float] = Field(
        default=None,
        description="止损价格",
    )
    tp_price: Optional[float] = Field(
        default=None,
        description="止盈价格",
    )
    confidence: float = Field(
        default=0.5,
        description="置信度 [0, 1]",
    )
    rationale: str = Field(
        default="",
        description="决策理由",
    )


def _prune_none(d: Dict) -> Dict:
    """Recursively remove None values from dict."""
    if not isinstance(d, dict):
        return d
    return {
        k: _prune_none(v) if isinstance(v, dict) else v
        for k, v in d.items()
        if v is not None
    }


def _group_features(features: List) -> Dict:
    """Group features by interval/type."""
    result = {"market_snapshot": []}
    for feat in features:
        meta = getattr(feat, "meta", {}) or {}
        source = meta.get("source", "market_snapshot")
        if source not in result:
            result[source] = []
        result[source].append(feat.model_dump(mode="json") if hasattr(feat, "model_dump") else feat)
    return result


def _extract_market_section(features: List) -> Dict:
    """Extract compact market data from features."""
    market = {}
    for feat in features:
        if isinstance(feat, dict):
            inst = feat.get("instrument", {})
            symbol = inst.get("symbol") if isinstance(inst, dict) else str(inst)
            values = feat.get("values", {})
        else:
            symbol = feat.instrument.symbol if hasattr(feat, "instrument") else None
            values = feat.values if hasattr(feat, "values") else {}

        if symbol:
            market[symbol] = values
    return market


class LlmComposer(BaseComposer):
    """使用 LangGraph/LangChain 的 LLM 驱动 Composer。

    核心流程:
    1. 根据组合上下文构建序列化提示
    2. 调用 LLM 获取 TradePlanProposal
    3. 将提案规范化为可执行的 TradeInstruction 对象
    """

    def __init__(
        self,
        request: UserRequest,
        *,
        default_slippage_bps: int = 25,
        quantity_precision: float = 1e-9,
    ) -> None:
        """Initialize LLM composer.

        Args:
            request: 用户的配置
            default_slippage_bps: 默认滑点（单位：bp）
            quantity_precision: 最小数量精度
        """
        self._request = request
        self._default_slippage_bps = default_slippage_bps
        self._quantity_precision = quantity_precision

        # 使用工厂函数创建 LLM
        self._llm = create_llm_from_config(self._request.llm_model_config)

        # JSON 输出解析器
        self._parser = JsonOutputParser()

    def _build_prompt_text(self) -> str:
        """Return resolved prompt text.

        Priority order:
        1. prompt_text (direct text, highest priority)
        2. template_id (load from template file)
        3. Default fallback

        custom_prompt is prepended if provided.
        """
        trading_config = self._request.trading_config
        custom = trading_config.custom_prompt
        prompt = trading_config.prompt_text
        template_id = trading_config.template_id

        # Resolve main prompt content
        main_prompt = None

        # Priority 1: Direct prompt_text
        if prompt:
            main_prompt = prompt
        # Priority 2: Load from template
        elif template_id:
            try:
                loader = get_template_loader()
                main_prompt = loader.load(template_id)
                cprint(f"Loaded template '{template_id}' for strategy", "white")
            except TemplateNotFoundError as e:
                cprint(f"Template not found: {e}, using default", "yellow")
                main_prompt = None

        # Fallback: simple symbol-based prompt
        if not main_prompt:
            symbols = ", ".join(trading_config.symbols)
            main_prompt = f"Compose trading instructions for symbols: {symbols}."

        # Prepend custom prompt if provided
        if custom:
            return f"{custom}\n\n{main_prompt}"

        return main_prompt

    async def compose(self, context: ComposeContext) -> ComposeResult:
        """Compose trading instructions from context."""
        prompt = self._build_llm_prompt(context)
        try:
            plan = await self._call_llm(prompt)
            if not plan.items:
                cprint(
                    f"LLM 返回空的执行计划 compose_id={context.compose_id} "
                    f"依据={plan.rationale}",
                    "white"
                )
                return ComposeResult(instructions=[], rationale=plan.rationale)
        except Exception as exc:
            cprint(f"LLM invocation failed: {exc}", "red")
            return ComposeResult(
                instructions=[],
                rationale=f"LLM invocation failed: {exc}",
            )

        normalized = self._normalize_plan(context, plan)
        return ComposeResult(instructions=normalized, rationale=plan.rationale)

    @staticmethod
    def _build_summary(context: ComposeContext) -> Dict:
        """Build portfolio summary with risk metrics."""
        pv = context.portfolio

        return {
            "active_positions": sum(
                1
                for snap in pv.positions.values()
                if abs(float(getattr(snap, "quantity", 0.0) or 0.0)) > 0.0
            ),
            "total_value": pv.total_value,
            "account_balance": pv.account_balance,
            "free_cash": pv.free_cash,
            "unrealized_pnl": pv.total_unrealized_pnl,
            "sharpe_ratio": context.digest.sharpe_ratio,
        }

    def _build_history_section(self, context: ComposeContext) -> Optional[Dict]:
        """根据近期决策和历史摘要构建历史章节。

        Args:
            context: 构建包含 recent_decisions 和 history_summaries 的上下文

        Returns:
            历史记录部分字典，如果没有历史记录，则为 None
        """
        has_history = (
            context.recent_decisions
            or context.pending_signals
            or context.history_summaries
        )
        if not has_history:
            return None

        history = {}

        # 历史摘要（长期记忆，先显示让LLM先了解历史背景）
        if context.history_summaries:
            summaries = []
            for s in context.history_summaries:
                entry = {
                    "cycles": s.get("cycle_range"),
                    "summary": s.get("content"),
                }
                # 添加统计信息
                stats = s.get("stats")
                if stats:
                    entry["stats"] = stats
                summaries.append(entry)
            history["historical_summaries"] = summaries

        # 最近决策（短期记忆）
        if context.recent_decisions:
            # 格式化最近决策，只保留关键信息
            recent = []
            for d in context.recent_decisions[-5:]:  # 最多5条
                entry = {}
                if "cycle" in d:
                    entry["cycle"] = d["cycle"]
                if "action" in d:
                    entry["action"] = d["action"]
                if "symbol" in d:
                    entry["symbol"] = d["symbol"]
                if "qty" in d:
                    entry["qty"] = d["qty"]
                if "executed" in d:
                    entry["executed"] = d["executed"]
                if "exec_price" in d:
                    entry["price"] = d["exec_price"]
                if "realized_pnl" in d:
                    entry["pnl"] = d["realized_pnl"]
                if "reason" in d:
                    entry["reason"] = d["reason"]
                recent.append(entry)
            history["recent_decisions"] = recent

        # 待观察信号
        if context.pending_signals:
            history["pending_signals"] = context.pending_signals

        return history if history else None

    def _build_llm_prompt(self, context: ComposeContext) -> str:
        """Build structured prompt for LLM decision-making."""
        pv = context.portfolio

        # Build components
        summary = self._build_summary(context)
        features = _group_features(context.features)
        market = _extract_market_section(features.get("market_snapshot", []))

        # Portfolio positions
        positions = [
            {
                "symbol": sym,
                "qty": float(snap.quantity),
                "unrealized_pnl": snap.unrealized_pnl,
                "entry_ts": snap.entry_ts,
            }
            for sym, snap in pv.positions.items()
            if abs(float(snap.quantity)) > 0
        ]

        # Constraints
        constraints = (
            pv.constraints.model_dump(mode="json", exclude_none=True)
            if pv.constraints
            else {}
        )

        # 构建历史记忆部分
        history = self._build_history_section(context)

        payload = _prune_none(
            {
                "strategy_prompt": self._build_prompt_text(),
                "summary": summary,
                "history": history,  # 注入历史决策
                "market": market,
                "features": features,
                "positions": positions,
                "constraints": constraints,
            }
        )

        # 提取交易对列表
        symbols = list(market.keys())

        instructions = (
            "阅读上下文并为每个交易对分别进行独立分析和决策。\n\n"
            "features.15m = 结构趋势（240 个周期），features.1m = 实时信号（180 个周期）"
            f"待分析交易对: {symbols}\n\n"
            "分析要求:\n"
            "1. 对每个交易对进行独立的技术分析，不要混合分析\n"
            "2. 每个交易对的 rationale 必须包含该交易对的完整分析过程:\n"
            "   - 当前价格和涨跌幅\n"
            "   - 技术指标分析 (EMA、MACD、RSI 等)\n"
            "   - 资金费率评估\n"
            "   - 持仓情况（如有）\n"
            "   - 入场/出场信号判断\n"
            "   - 最终决策理由\n"
            "3. 即使选择 noop，也要在 rationale 中说明为何不操作\n\n"
            "上下文说明:\n"
            "- history.historical_summaries = 历史决策摘要（长期记忆）\n"
            "- history.recent_decisions = 最近的决策和执行结果（短期记忆）\n"
            "- features.market_snapshot = 当前价格和指标\n"
            "- market.funding.rate: 正值表示多头付给空头\n\n"
            "遵守约束条件。输出包含 items 数组的 JSON，每个交易对一个 item。"
        )

        return f"{instructions}\n\nContext:\n{json.dumps(payload, ensure_ascii=False)}"

    async def _call_llm(self, prompt: str) -> TradePlanProposal:
        """调用 LLM 获取结构化交易计划。

        使用 JsonOutputParser 解析 JSON 输出（兼容 DeepSeek 等不支持 response_format 的模型）。
        """
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            # 使用普通调用 + JSON 解析（兼容性更好）
            response = await self._llm.ainvoke(messages)

            # 提取内容
            content = response.content if hasattr(response, 'content') else str(response)

            # 尝试从 markdown 代码块中提取 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                # 可能是没有语言标记的代码块
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1].strip()

            # 解析 JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # 尝试查找 JSON 对象
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    cprint(f"Cannot find JSON in response: {content[:500]}", "red")
                    return TradePlanProposal(items=[], rationale="Cannot parse JSON from response")

            # 构建 TradePlanProposal
            items = []
            for item_data in data.get("items", []):
                try:
                    items.append(TradeDecisionItem(**item_data))
                except Exception as e:
                    cprint(f"Failed to parse item: {item_data}, error: {e}", "yellow")

            result = TradePlanProposal(
                ts=get_current_timestamp_ms(),
                items=items,
                rationale=data.get("rationale"),
            )

            # 打印 LLM 决策结果
            self._print_decision_result(result)

            return result

        except ValidationError as e:
            cprint(f"Validation error parsing LLM response: {e}", "red")
            return TradePlanProposal(
                items=[],
                rationale=f"Validation error: {e}",
            )
        except Exception as e:
            cprint(f"Error calling LLM: {e}", "red")
            raise

    def _print_decision_result(self, result: TradePlanProposal) -> None:
        """打印格式化的 LLM 决策结果."""
        # 动作类型映射到中文和颜色
        action_display = {
            "open_long": ("开多", "green"),
            "open_short": ("开空", "red"),
            "close_long": ("平多", "yellow"),
            "close_short": ("平空", "yellow"),
            "hold": ("持有", "white"),
        }

        cprint("\n" + "=" * 70, "cyan")
        cprint("                    📊 LLM 决策结果", "cyan", attrs=["bold"])
        cprint("=" * 70, "cyan")

        if not result.items:
            cprint("  (无交易指令)", "white")
        else:
            for i, item in enumerate(result.items, 1):
                symbol = item.instrument.symbol
                action_val = item.action.value if hasattr(item.action, 'value') else str(item.action)
                action_cn, action_color = action_display.get(action_val, (action_val, "white"))

                # 标题行
                cprint(f"\n  [{i}] {symbol}", "white", attrs=["bold"])
                cprint("  " + "-" * 40, "white")

                # 动作和置信度
                confidence_pct = f"{(item.confidence or 0) * 100:.0f}%" if item.confidence else "N/A"
                cprint(f"      动作: {action_cn:<8}  置信度: {confidence_pct}", action_color)

                # 数量和杠杆
                qty_str = f"{item.target_qty:.6f}".rstrip('0').rstrip('.') if item.target_qty else "N/A"
                leverage_str = f"{item.leverage:.0f}x" if item.leverage else "N/A"
                cprint(f"      数量: {qty_str:<12}  杠杆: {leverage_str}", "white")

                # 止损止盈
                sl_str = f"{item.sl_price:.4f}" if item.sl_price else "未设置"
                tp_str = f"{item.tp_price:.4f}" if item.tp_price else "未设置"
                cprint(f"      止损: {sl_str:<12}  止盈: {tp_str}", "white")

                # 理由
                if item.rationale:
                    rationale = item.rationale.strip()
                    # 自动换行显示完整理由
                    cprint("      理由:", "white")
                    # 按行分割并缩进显示
                    lines = self._wrap_text(rationale, width=60)
                    for line in lines:
                        cprint(f"        {line}", "white")

        # 整体决策理由
        cprint("\n" + "-" * 70, "cyan")
        cprint("  📝 整体决策:", "cyan", attrs=["bold"])
        if result.rationale:
            lines = self._wrap_text(result.rationale.strip(), width=65)
            for line in lines:
                cprint(f"     {line}", "white")
        else:
            cprint("     (无)", "white")

        cprint("=" * 70 + "\n", "cyan")

    def _wrap_text(self, text: str, width: int = 60) -> List[str]:
        """将长文本按指定宽度换行."""
        lines = []
        # 先按换行符分割
        for paragraph in text.split('\n'):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            # 按宽度分割（简单实现，按字符数）
            while len(paragraph) > width:
                # 找一个合适的断点
                break_point = width
                # 尝试在空格或标点处断开
                for i in range(width, max(width - 20, 0), -1):
                    if paragraph[i] in ' ，。、；：！？,.;:!? ':
                        break_point = i + 1
                        break
                lines.append(paragraph[:break_point].strip())
                paragraph = paragraph[break_point:].strip()
            if paragraph:
                lines.append(paragraph)
        return lines if lines else ["(无)"]

    # 默认止损百分比（用于开仓时 LLM 未给出止损价的情况）
    DEFAULT_SL_PCT = 0.02  # 2%
    # 止损范围限制
    MIN_SL_PCT = 0.005  # 0.5%
    MAX_SL_PCT = 0.10   # 10%

    def _validate_sl_price(
        self,
        sl_price: Optional[float],
        entry_price: float,
        is_long: bool,
    ) -> Optional[float]:
        """Validate and normalize stop loss price.

        Args:
            sl_price: LLM-provided stop loss price (may be None or invalid)
            entry_price: Expected entry price
            is_long: True for long position, False for short

        Returns:
            Validated sl_price, or default if invalid/missing
        """
        if entry_price <= 0:
            return None

        # 计算默认止损价
        if is_long:
            default_sl = entry_price * (1 - self.DEFAULT_SL_PCT)
        else:
            default_sl = entry_price * (1 + self.DEFAULT_SL_PCT)

        if sl_price is None:
            cprint(f"No sl_price provided, using default: {default_sl:.6f}", "white")
            return default_sl

        # 验证方向正确性
        if is_long and sl_price >= entry_price:
            cprint(
                f"Invalid sl_price for long: {sl_price} >= entry {entry_price}, using default",
                "yellow"
            )
            return default_sl
        if not is_long and sl_price <= entry_price:
            cprint(
                f"Invalid sl_price for short: {sl_price} <= entry {entry_price}, using default",
                "yellow"
            )
            return default_sl

        # 验证止损幅度在合理范围内
        sl_pct = abs(sl_price - entry_price) / entry_price
        if sl_pct < self.MIN_SL_PCT:
            cprint(
                f"sl_price too tight ({sl_pct:.2%} < {self.MIN_SL_PCT:.2%}), using default",
                "yellow"
            )
            return default_sl
        if sl_pct > self.MAX_SL_PCT:
            cprint(
                f"sl_price too wide ({sl_pct:.2%} > {self.MAX_SL_PCT:.2%}), clamping",
                "yellow"
            )
            if is_long:
                return entry_price * (1 - self.MAX_SL_PCT)
            else:
                return entry_price * (1 + self.MAX_SL_PCT)

        return sl_price

    def _validate_tp_price(
        self,
        tp_price: Optional[float],
        entry_price: float,
        sl_price: Optional[float],
        is_long: bool,
    ) -> Optional[float]:
        """Validate take profit price.

        Args:
            tp_price: LLM-provided take profit price (may be None)
            entry_price: Expected entry price
            sl_price: Validated stop loss price
            is_long: True for long position, False for short

        Returns:
            Validated tp_price, or None if invalid/not provided
        """
        if tp_price is None or entry_price <= 0:
            return None

        # 验证方向正确性
        if is_long and tp_price <= entry_price:
            cprint(
                f"Invalid tp_price for long: {tp_price} <= entry {entry_price}, ignoring",
                "yellow"
            )
            return None
        if not is_long and tp_price >= entry_price:
            cprint(
                f"Invalid tp_price for short: {tp_price} >= entry {entry_price}, ignoring",
                "yellow"
            )
            return None

        # 可选：验证盈亏比至少 1:1
        if sl_price:
            risk = abs(entry_price - sl_price)
            reward = abs(tp_price - entry_price)
            if reward < risk:
                cprint(
                    f"tp/sl ratio {reward/risk:.2f} < 1, consider adjusting",
                    "yellow"
                )

        return tp_price

    def _normalize_plan(
        self,
        context: ComposeContext,
        plan: TradePlanProposal,
    ) -> List[TradeInstruction]:
        """Normalize plan into executable instructions.

        Applies guardrails:
        - Quantity step/min/max constraints
        - Buying power limits
        - Notional caps
        - Stop loss / take profit validation
        """
        instructions: List[TradeInstruction] = []
        pv = context.portfolio
        constraints = pv.constraints

        # Track available buying power
        available_bp = float(pv.buying_power or pv.free_cash or 0.0)

        for item in plan.items:
            # Skip noop actions
            if item.action == TradeDecisionAction.NOOP:
                continue

            symbol = item.instrument.symbol
            qty = float(item.target_qty)
            leverage = float(item.leverage or 1.0)

            # For close operations with zero/small qty, use current position quantity
            if item.action in (TradeDecisionAction.CLOSE_LONG, TradeDecisionAction.CLOSE_SHORT):
                current_pos = pv.positions.get(symbol)
                if current_pos and abs(current_pos.quantity) > 0:
                    pos_qty = abs(current_pos.quantity)
                    if qty <= 0 or qty < pos_qty * 0.01:  # qty is 0 or negligible
                        cprint(
                            f"{symbol}: 平仓操作 qty={qty} 太小，使用当前持仓量 {pos_qty}",
                            "white"
                        )
                        qty = pos_qty
                else:
                    cprint(f"{symbol}: 平仓操作但无持仓，跳过", "yellow")
                    continue

            # Apply quantity constraints
            if constraints:
                if constraints.min_trade_qty and qty < constraints.min_trade_qty:
                    cprint(
                        f"Skipping {symbol}: qty {qty} < min {constraints.min_trade_qty}",
                        "yellow"
                    )
                    continue
                if constraints.max_order_qty and qty > constraints.max_order_qty:
                    qty = constraints.max_order_qty
                if constraints.quantity_step:
                    qty = round(qty / constraints.quantity_step) * constraints.quantity_step
                if constraints.max_leverage and leverage > constraints.max_leverage:
                    leverage = constraints.max_leverage

            # Estimate notional and check buying power
            # Get price from features
            price = None
            for feat in context.features:
                if feat.instrument.symbol == symbol:
                    values = feat.values or {}
                    # 尝试多种价格字段
                    price = (
                        values.get("price.last")
                        or values.get("price.close")
                        or values.get("close")
                        or values.get("last")
                    )
                    if price:
                        break

            if not price:
                cprint(f"{symbol}: 无法从 features 获取价格，止损将无法设置", "yellow")

            if price:
                notional = qty * float(price)
                margin_required = notional / leverage

                # Cap factor limit
                cap = self._request.trading_config.cap_factor
                max_notional = float(pv.total_value or pv.account_balance) * cap
                if notional > max_notional:
                    qty = max_notional / float(price)
                    notional = qty * float(price)
                    margin_required = notional / leverage

                # Buying power check for opening positions
                if item.action in (TradeDecisionAction.OPEN_LONG, TradeDecisionAction.OPEN_SHORT):
                    if margin_required > available_bp:
                        # Reduce size to fit
                        qty = (available_bp * leverage) / float(price) * 0.95  # 5% buffer
                        if qty <= 0:
                            cprint(
                                f"Skipping {symbol}: insufficient buying power",
                                "yellow"
                            )
                            continue
                        margin_required = (qty * float(price)) / leverage

                    available_bp -= margin_required

            # Skip if quantity too small after adjustments
            if qty < self._quantity_precision:
                cprint(f"Skipping {symbol}: qty too small after adjustments", "yellow")
                continue

            # Re-check min_trade_qty after all adjustments (quantity_step, cap, etc.)
            if constraints and constraints.min_trade_qty and qty < constraints.min_trade_qty:
                cprint(
                    f"Skipping {symbol}: adjusted qty {qty} < min_trade_qty {constraints.min_trade_qty}",
                    "yellow"
                )
                continue

            # Check minimum notional value (most exchanges require ~$5-10 min notional)
            if price:
                notional = qty * float(price)
                min_notional = (
                    constraints.min_notional if constraints and constraints.min_notional else 5.0
                )
                if notional < min_notional:
                    # Try to bump up to minimum notional
                    min_qty_for_notional = min_notional / float(price) * 1.05  # 5% buffer
                    # Re-apply quantity step if needed
                    if constraints and constraints.quantity_step:
                        min_qty_for_notional = (
                            round(min_qty_for_notional / constraints.quantity_step + 0.5)
                            * constraints.quantity_step
                        )

                    # Only check buying power for opening positions
                    is_opening = item.action in (
                        TradeDecisionAction.OPEN_LONG, TradeDecisionAction.OPEN_SHORT
                    )
                    if is_opening:
                        # Check if we can afford the bumped quantity
                        # Note: available_bp was already reduced above for this position
                        original_margin = (qty * float(price)) / leverage
                        bumped_margin = (min_qty_for_notional * float(price)) / leverage
                        # Add back what we already deducted for this position
                        effective_available = available_bp + original_margin
                        if bumped_margin <= effective_available * 0.95:  # Leave some buffer
                            cprint(
                                f"{symbol}: bumping qty from {qty} to {min_qty_for_notional} "
                                f"to meet min_notional {min_notional}",
                                "white"
                            )
                            qty = min_qty_for_notional
                            # Adjust available_bp: add back original deduction, deduct new amount
                            available_bp = effective_available - bumped_margin
                        else:
                            cprint(
                                f"Skipping {symbol}: notional {notional:.2f} < min_notional {min_notional}, "
                                f"insufficient buying power to bump up",
                                "yellow"
                            )
                            continue
                    else:
                        # For closing positions, just bump up the qty (no margin needed)
                        cprint(
                            f"{symbol}: bumping close qty from {qty} to {min_qty_for_notional} "
                            f"to meet min_notional {min_notional}",
                            "white"
                        )
                        qty = min_qty_for_notional

            # Derive side from action
            side = derive_side_from_action(item.action)
            if side is None:
                continue

            # Process stop loss / take profit for opening positions
            sl_price = None
            tp_price = None
            if item.action in (TradeDecisionAction.OPEN_LONG, TradeDecisionAction.OPEN_SHORT):
                is_long = item.action == TradeDecisionAction.OPEN_LONG
                entry_price = float(price) if price else 0.0

                # Validate and get sl_price (always set for opens)
                sl_price = self._validate_sl_price(
                    item.sl_price, entry_price, is_long
                )

                # Validate tp_price (optional)
                tp_price = self._validate_tp_price(
                    item.tp_price, entry_price, sl_price, is_long
                )

                if sl_price:
                    tp_str = f"{tp_price:.2f}" if tp_price else "N/A"
                    cprint(
                        f"{symbol} {'LONG' if is_long else 'SHORT'}: "
                        f"entry≈{entry_price:.2f}, sl={sl_price:.2f}, tp={tp_str}",
                        "white"
                    )

            # Create instruction
            inst = TradeInstruction(
                instruction_id=f"{context.compose_id}:{symbol}",
                compose_id=context.compose_id,
                instrument=item.instrument,
                action=item.action,
                side=side,
                quantity=qty,
                leverage=leverage,
                price_mode=PriceMode.MARKET,
                max_slippage_bps=self._default_slippage_bps,
                sl_price=sl_price,
                tp_price=tp_price,
                meta={
                    "confidence": item.confidence,
                    "rationale": item.rationale,
                },
            )
            instructions.append(inst)

        return instructions
