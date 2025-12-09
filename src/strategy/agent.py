"""策略代理模块。

负责编排交易循环:
1. 获取市场数据
2. 通过 LLM 进行决策
3. 执行交易
4. 更新仓位和历史

支持反思模式(Reflection Mode):
- 自动分析历史交易表现
- 识别问题模式并生成改进建议
- 根据夏普比等指标自动调整行为
"""

import asyncio
from typing import Optional

from termcolor import cprint

from src.trading import (
    DecisionCycleResult,
    StopReason,
    StrategyStatus,
    UserRequest,
)
from src.trading.engine import StrategyRuntime, create_strategy_runtime
from src.trading.decision import BaseComposer, LlmComposer
from src.trading.market import BaseFeaturesPipeline, DefaultFeaturesPipeline


class StrategyAgent:
    """策略代理，基于 LLM 进行交易决策。

    交易循环:
    1. 初始化运行时组件
    2. 按配置间隔运行决策周期
    3. 处理优雅关闭和平仓

    反思模式(enable_reflection=True):
    - 分析历史交易表现(夏普比、胜率、回撤等)
    - 识别问题模式(过度交易、连续亏损等)
    - 自动调整置信度阈值和交易频率
    - 在严重问题时触发冷静期
    """

    def __init__(
        self,
        request: UserRequest,
        composer: Optional[BaseComposer] = None,
        features_pipeline: Optional[BaseFeaturesPipeline] = None,
        *,
        enable_reflection: bool = False,
    ) -> None:
        """初始化策略代理。

        Args:
            request: 用户请求配置
            composer: 自定义决策器
            features_pipeline: 自定义特征管道
            enable_reflection: 是否启用反思模式
        """
        self._request = request
        self._composer = composer
        self._features_pipeline = features_pipeline
        self._enable_reflection = enable_reflection
        self._runtime: Optional[StrategyRuntime] = None
        self._running = False
        self._stop_reason: Optional[StopReason] = None

    async def _build_features_pipeline(self) -> Optional[BaseFeaturesPipeline]:
        """构建特征管道。"""
        if self._features_pipeline:
            return self._features_pipeline
        return DefaultFeaturesPipeline.from_request(self._request)

    def _create_decision_composer(self) -> Optional[BaseComposer]:
        """构建决策器。"""
        if self._composer:
            return self._composer

        if self._enable_reflection:
            from src.trading.reflection import ReflectiveComposer

            cprint("启用反思模式(Reflection Mode)", "white")
            return ReflectiveComposer(
                request=self._request,
                enable_cooldown=True,
                enable_confidence_filter=True,
                enable_symbol_filter=True,
            )

        return LlmComposer(request=self._request)

    async def start(self) -> str:
        """启动策略代理。

        Returns:
            策略 ID
        """
        composer = self._create_decision_composer()
        features_pipeline = await self._build_features_pipeline()

        self._runtime = await create_strategy_runtime(
            self._request,
            composer=composer,
            features_pipeline=features_pipeline,
            strategy_id_override=self._request.trading_config.strategy_id,
        )

        # 反思模式: 注入 history_recorder
        if self._enable_reflection:
            from src.trading.reflection import ReflectiveComposer

            if isinstance(composer, ReflectiveComposer) and self._runtime.history_recorder:
                composer.set_history_recorder(self._runtime.history_recorder)
                cprint("反思模式已连接历史记录器", "white")

        strategy_id = self._runtime.strategy_id
        cprint(f"策略已启动: {strategy_id}", "green")

        self._running = True
        return strategy_id

    async def run(self) -> None:
        """运行主交易循环。"""
        if not self._runtime:
            await self.start()

        strategy_id = self._runtime.strategy_id
        interval = self._request.trading_config.decide_interval

        cprint(f"开始决策循环: {strategy_id}", "white")

        try:
            while self._running:
                result = await self._runtime.run_cycle()
                cprint(
                    f"周期 {result.cycle_index} 完成: "
                    f"{len(result.trades)} 笔交易, "
                    f"盈亏: {result.strategy_summary.realized_pnl:.2f}",
                    "white"
                )

                for inst in result.instructions:
                    cprint(
                        f"  -> {inst.action.value if inst.action else 'N/A'} "
                        f"{inst.instrument.symbol} 数量={inst.quantity}",
                        "white"
                    )

                # 等待下一个循环
                for _ in range(interval):
                    if not self._running:
                        break
                    await asyncio.sleep(1)

            cprint(f"策略已停止: {strategy_id}", "white")
            self._stop_reason = StopReason.NORMAL_EXIT

        except asyncio.CancelledError:
            self._stop_reason = StopReason.CANCELLED
            cprint(f"策略已取消: {strategy_id}", "yellow")
            raise
        except Exception as e:
            self._stop_reason = StopReason.ERROR
            cprint(f"策略错误: {e}", "red"); import traceback; traceback.print_exc()
            raise
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """优雅停止策略。"""
        cprint("正在停止策略...", "white")
        self._running = False

    async def _cleanup(self) -> None:
        """清理资源并平仓。"""
        if self._runtime:
            try:
                trades = await self._runtime.coordinator.close_all_positions()
                if trades:
                    cprint(f"关闭时平仓 {len(trades)} 个持仓")
            except Exception:
                cprint("平仓时出错", "red"); import traceback; traceback.print_exc()
                self._stop_reason = StopReason.ERROR_CLOSING_POSITIONS

            try:
                await self._runtime.coordinator.close()
            except Exception:
                cprint("关闭协调器时出错", "red"); import traceback; traceback.print_exc()

    @property
    def strategy_id(self) -> Optional[str]:
        """获取当前策略 ID。"""
        return self._runtime.strategy_id if self._runtime else None

    @property
    def is_running(self) -> bool:
        """检查是否正在运行。"""
        return self._running

    @property
    def status(self) -> StrategyStatus:
        """获取当前状态。"""
        return StrategyStatus.RUNNING if self._running else StrategyStatus.STOPPED

    @property
    def stop_reason(self) -> Optional[StopReason]:
        """获取停止原因。"""
        return self._stop_reason


async def run_strategy(request: UserRequest) -> None:
    """便捷函数：运行策略。

    Args:
        request: 用户请求配置
    """
    agent = StrategyAgent(request)
    await agent.start()
    await agent.run()
