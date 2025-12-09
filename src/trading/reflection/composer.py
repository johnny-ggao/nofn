"""反思型决策器。

在标准 LlmComposer 基础上增加反思能力：
1. 在决策前进行反思分析
2. 将反思洞察注入提示词
3. 根据反思建议调整行为
"""

from typing import List, Optional

from termcolor import cprint

from ..decision.interfaces import BaseComposer
from ..decision.llm_composer import LlmComposer
from ..history.recorder import InMemoryHistoryRecorder
from ..models import (
    ComposeContext,
    ComposeResult,
    HistoryRecord,
    TradeDecisionAction,
    TradeInstruction,
    UserRequest,
)
from .analyzer import ReflectionAnalyzer
from .models import ReflectionInsight


class ReflectiveComposer(BaseComposer):
    """带反思能力的决策器。

    在每次决策前：
    1. 调用 ReflectionAnalyzer 分析历史
    2. 将反思洞察注入提示词
    3. 根据建议过滤或调整决策

    用法:
        composer = ReflectiveComposer(request=request)
        composer.set_history_recorder(recorder)  # 运行时注入
        result = await composer.compose(context)
    """

    def __init__(
        self,
        request: UserRequest,
        *,
        enable_cooldown: bool = True,
        enable_confidence_filter: bool = True,
        enable_symbol_filter: bool = True,
    ) -> None:
        """初始化反思型决策器。

        Args:
            request: 用户请求配置
            enable_cooldown: 是否启用冷静期(严重问题时暂停交易)
            enable_confidence_filter: 是否根据反思建议过滤低置信度交易
            enable_symbol_filter: 是否根据反思建议过滤表现差的标的
        """
        self._request = request
        self._history_recorder: Optional[InMemoryHistoryRecorder] = None

        # 功能开关
        self._enable_cooldown = enable_cooldown
        self._enable_confidence_filter = enable_confidence_filter
        self._enable_symbol_filter = enable_symbol_filter

        # 内部组件
        self._base_composer = LlmComposer(request)
        self._analyzer = ReflectionAnalyzer()

        # 状态
        self._last_insight: Optional[ReflectionInsight] = None
        self._cooldown_remaining: int = 0

    def set_history_recorder(self, recorder: InMemoryHistoryRecorder) -> None:
        """设置历史记录器(运行时注入)。"""
        self._history_recorder = recorder

    async def compose(self, context: ComposeContext) -> ComposeResult:
        """执行带反思的决策。

        流程:
        1. 检查是否在冷静期
        2. 执行反思分析
        3. 注入反思上下文到提示词
        4. 调用基础决策器
        5. 根据反思建议过滤结果
        """
        # 步骤1: 检查冷静期
        if self._enable_cooldown and self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            cprint(
                f"反思建议冷静期中，剩余 {self._cooldown_remaining + 1} 个周期，选择 noop",
                "cyan"
            )
            return ComposeResult(
                instructions=[],
                rationale=f"反思建议暂停交易，冷静期剩余 {self._cooldown_remaining + 1} 个周期",
            )

        # 步骤2: 执行反思分析
        insight = self._perform_reflection(context)
        self._last_insight = insight

        # 检查是否需要进入冷静期
        if self._enable_cooldown and insight.cooldown_cycles > 0:
            self._cooldown_remaining = insight.cooldown_cycles - 1
            cprint(
                f"反思触发冷静期: {insight.cooldown_cycles} 个周期。原因: {insight.summary}",
                "yellow"
            )
            return ComposeResult(
                instructions=[],
                rationale=f"反思触发冷静期: {insight.summary}",
            )

        # 步骤3: 注入反思上下文
        enhanced_context = self._inject_reflection_context(context, insight)

        # 步骤4: 调用基础决策器
        result = await self._base_composer.compose(enhanced_context)

        # 步骤5: 根据反思建议过滤结果
        filtered_instructions = self._apply_reflection_filters(
            result.instructions, insight
        )

        # 如果有指令被过滤，更新 rationale
        if len(filtered_instructions) < len(result.instructions):
            filtered_count = len(result.instructions) - len(filtered_instructions)
            rationale = (
                f"{result.rationale or ''} "
                f"[反思过滤: {filtered_count} 条指令因不符合反思建议被移除]"
            ).strip()
        else:
            rationale = result.rationale

        return ComposeResult(
            instructions=filtered_instructions,
            rationale=rationale,
        )

    def _perform_reflection(self, context: ComposeContext) -> ReflectionInsight:
        """执行反思分析。"""
        # 获取历史记录
        records: List[HistoryRecord] = []
        if self._history_recorder:
            records = self._history_recorder.get_records()

        # 执行分析
        insight = self._analyzer.analyze(
            digest=context.digest,
            recent_records=records,
            previous_insight=self._last_insight,
        )

        # 记录反思结果
        if insight.alerts:
            for alert in insight.alerts:
                log_fn = lambda msg: cprint(msg, "yellow") if alert.severity == "critical" else cprint(msg, "white")
                log_fn(f"反思警报 [{alert.severity}]: {alert.message}")

        if insight.lessons:
            cprint(f"反思教训: {[l.lesson for l in insight.lessons[:2]]}", "white")

        return insight

    def _inject_reflection_context(
        self, context: ComposeContext, insight: ReflectionInsight
    ) -> ComposeContext:
        """将反思洞察注入到上下文中。

        通过修改 trading_config.custom_prompt 来注入反思内容。
        """
        # 生成反思上下文文本
        reflection_text = insight.to_prompt_context()

        # 获取原有的 custom_prompt
        original_custom = self._request.trading_config.custom_prompt or ""

        # 拼接反思内容
        enhanced_custom = f"{original_custom}\n\n{reflection_text}".strip()

        # 创建增强的请求(不修改原始请求)
        # 这里我们直接修改 context 无法做到，所以通过另一种方式：
        # 在 ComposeContext 的 meta 中添加反思信息
        # 基础 composer 会通过 strategy_prompt 使用它

        # 实际上，我们需要修改 _build_prompt_text 的行为
        # 更简单的方式是：临时修改 request 的 custom_prompt
        # 但这不够优雅，我们采用在 meta 中传递的方式

        # 由于 ComposeContext 是 frozen，我们需要创建新的
        # 但 ComposeContext 没有 meta 字段，所以我们需要另一种方式

        # 最简单的方式：临时设置 custom_prompt
        self._request.trading_config.custom_prompt = enhanced_custom

        return context

    def _apply_reflection_filters(
        self,
        instructions: List[TradeInstruction],
        insight: ReflectionInsight,
    ) -> List[TradeInstruction]:
        """根据反思建议过滤指令。"""
        filtered: List[TradeInstruction] = []

        for inst in instructions:
            # 跳过 NOOP
            if inst.action == TradeDecisionAction.NOOP:
                filtered.append(inst)
                continue

            symbol = inst.instrument.symbol

            # 过滤器1: 标的黑名单
            if self._enable_symbol_filter and symbol in insight.symbols_to_avoid:
                cprint(
                    f"反思过滤: {symbol} 在回避列表中，跳过 {inst.action.value}",
                    "cyan"
                )
                continue

            # 过滤器2: 置信度门槛
            if self._enable_confidence_filter and insight.suggested_min_confidence:
                meta = inst.meta or {}
                confidence = meta.get("confidence", 1.0)
                if confidence < insight.suggested_min_confidence:
                    cprint(
                        f"反思过滤: {symbol} 置信度 {confidence:.2f} "
                        f"低于建议阈值 {insight.suggested_min_confidence:.2f}",
                        "cyan"
                    )
                    continue

            filtered.append(inst)

        return filtered

    @property
    def last_insight(self) -> Optional[ReflectionInsight]:
        """获取最近一次反思洞察。"""
        return self._last_insight

    @property
    def cooldown_remaining(self) -> int:
        """获取剩余冷静期周期数。"""
        return self._cooldown_remaining

    def reset_cooldown(self) -> None:
        """重置冷静期。"""
        self._cooldown_remaining = 0
        cprint("反思冷静期已手动重置", "white")
