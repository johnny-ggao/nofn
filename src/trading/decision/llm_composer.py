"""LLM-driven composer using LangGraph (replacing Agno).

This module provides the LlmComposer that uses LangChain/LangGraph for
structured LLM outputs instead of Agno.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import ValidationError

from ..models import (
    ComposeContext,
    ComposeResult,
    InstrumentRef,
    PriceMode,
    TradeDecisionAction,
    TradeDecisionItem,
    TradeInstruction,
    TradePlanProposal,
    TradeSide,
    UserRequest,
    derive_side_from_action,
    get_current_timestamp_ms,
)
from .interfaces import BaseComposer
from .system_prompt import SYSTEM_PROMPT

# Import template loader
from ..templates import (
    TemplateNotFoundError,
    get_template_loader,
)


def _create_llm(
    provider: str,
    model_id: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.4,
) -> ChatOpenAI:
    """Create LLM instance based on provider.

    Supports:
    - openai: Direct OpenAI API
    - openrouter: OpenRouter API (OpenAI-compatible)
    - deepseek: DeepSeek API
    - qwen/dashscope: Alibaba Qwen API

    Args:
        provider: Model provider name
        model_id: Model identifier
        api_key: API key (falls back to environment variables)
        base_url: Custom API base URL (falls back to provider defaults)
        temperature: Model temperature
    """
    # Resolve API key from environment if not provided
    if not api_key:
        env_key_map = {
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_key = env_key_map.get(provider.lower())
        if env_key:
            api_key = os.getenv(env_key)

    if not api_key:
        raise ValueError(
            f"API key not provided for provider '{provider}'. "
            f"Set it in config or via environment variable."
        )

    # Configure base URL - use provided value or fall back to provider defaults
    if not base_url:
        default_urls = {
            "openrouter": "https://openrouter.ai/api/v1",
            "deepseek": "https://api.deepseek.com",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        base_url = default_urls.get(provider.lower())

    # Create LLM
    kwargs: Dict[str, Any] = {
        "model": model_id,
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


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

        cfg = self._request.llm_model_config
        self._llm = _create_llm(
            provider=cfg.provider,
            model_id=cfg.model_id,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
        )

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
                logger.debug(f"Loaded template '{template_id}' for strategy")
            except TemplateNotFoundError as e:
                logger.warning(f"Template not found: {e}, using default")
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
                logger.info(
                    f"LLM returned empty plan for compose_id={context.compose_id} "
                    f"with rationale={plan.rationale}"
                )
                return ComposeResult(instructions=[], rationale=plan.rationale)
        except Exception as exc:
            logger.error(f"LLM invocation failed: {exc}")
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
        """Build history section from recent decisions and historical summaries.

        Args:
            context: Compose context containing recent_decisions and history_summaries

        Returns:
            History section dict or None if no history
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

        instructions = (
            "阅读上下文并做出决策。"
            "history.historical_summaries = 历史决策摘要（长期记忆），了解过去的决策模式和教训。"
            "history.recent_decisions = 最近的决策和执行结果（短期记忆），参考它保持决策连贯性。"
            "features.market_snapshot = 当前价格和指标。"
            "market.funding.rate：正值表示多头付给空头。"
            "遵守约束条件和风险标志。信号不明确时优先选择 noop（不操作）。"
            "始终在顶层包含简洁的 rationale（决策理由）。"
            "若选择 noop（items 为空），需在 rationale 中说明原因。"
            "按照输出格式要求，输出包含 items 数组的 JSON。"
        )

        return f"{instructions}\n\nContext:\n{json.dumps(payload, ensure_ascii=False)}"

    async def _call_llm(self, prompt: str) -> TradePlanProposal:
        """Invoke LLM and parse response into TradePlanProposal."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self._llm.ainvoke(messages)
            content = response.content

            # Parse JSON from response
            if isinstance(content, str):
                # Try to extract JSON from response
                try:
                    # Handle markdown code blocks
                    if "```json" in content:
                        start = content.find("```json") + 7
                        end = content.find("```", start)
                        content = content[start:end].strip()
                    elif "```" in content:
                        start = content.find("```") + 3
                        end = content.find("```", start)
                        content = content[start:end].strip()

                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    # Try to find JSON object in text
                    import re
                    match = re.search(r'\{[\s\S]*\}', content)
                    if match:
                        parsed = json.loads(match.group())
                    else:
                        logger.error(f"Could not parse JSON from LLM response: {content[:200]}")
                        return TradePlanProposal(
                            items=[],
                            rationale=f"Failed to parse LLM response as JSON",
                        )

                # Validate and convert to TradePlanProposal
                return TradePlanProposal(
                    ts=get_current_timestamp_ms(),
                    items=[
                        TradeDecisionItem(**item)
                        for item in parsed.get("items", [])
                    ],
                    rationale=parsed.get("rationale"),
                )
            else:
                logger.error(f"Unexpected LLM response type: {type(content)}")
                return TradePlanProposal(items=[], rationale="Unexpected response type")

        except ValidationError as e:
            logger.error(f"Validation error parsing LLM response: {e}")
            return TradePlanProposal(
                items=[],
                rationale=f"Validation error: {e}",
            )
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise

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

            # Apply quantity constraints
            if constraints:
                if constraints.min_trade_qty and qty < constraints.min_trade_qty:
                    logger.warning(
                        f"Skipping {symbol}: qty {qty} < min {constraints.min_trade_qty}"
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
                    price = values.get("price.last") or values.get("price.close")
                    break

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
                            logger.warning(
                                f"Skipping {symbol}: insufficient buying power"
                            )
                            continue
                        margin_required = (qty * float(price)) / leverage

                    available_bp -= margin_required

            # Skip if quantity too small after adjustments
            if qty < self._quantity_precision:
                logger.warning(f"Skipping {symbol}: qty too small after adjustments")
                continue

            # Derive side from action
            side = derive_side_from_action(item.action)
            if side is None:
                continue

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
                meta={
                    "confidence": item.confidence,
                    "rationale": item.rationale,
                },
            )
            instructions.append(inst)

        return instructions
