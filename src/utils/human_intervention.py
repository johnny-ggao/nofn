"""
人工介入管理器

提供多种人工介入模式，让人类专家可以在系统运行时：
- 审批/否决交易决策
- 暂停/恢复系统运行
- 实时调整策略参数
- 覆盖LLM决策
- 提供反馈用于学习
"""
import asyncio
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from datetime import datetime
from termcolor import cprint
import json


class InterventionMode(Enum):
    """人工介入模式"""
    DISABLED = "disabled"           # 完全自动，无人工介入
    REVIEW_ONLY = "review_only"     # 仅展示，不阻塞
    APPROVE_TRADES = "approve_trades"  # 交易需人工批准
    APPROVE_ALL = "approve_all"     # 所有决策需批准
    INTERACTIVE = "interactive"     # 完全交互模式


class InterventionPoint(Enum):
    """介入点类型"""
    BEFORE_DECISION = "before_decision"      # 决策前
    AFTER_DECISION = "after_decision"        # 决策后，执行前
    AFTER_EXECUTION = "after_execution"      # 执行后
    BEFORE_EVALUATION = "before_evaluation"  # 评估前


class HumanInterventionManager:
    """
    人工介入管理器

    使用方式：
    ```python
    manager = HumanInterventionManager(mode=InterventionMode.APPROVE_TRADES)

    # 在决策后，执行前介入
    decision = await manager.intervene(
        point=InterventionPoint.AFTER_DECISION,
        data={"decision": decision, "market": market_snapshot},
        allow_override=True
    )
    ```
    """

    def __init__(
        self,
        mode: InterventionMode = InterventionMode.DISABLED,
        timeout_seconds: int = 60,
        auto_approve_on_timeout: bool = True,
    ):
        """
        初始化人工介入管理器

        Args:
            mode: 介入模式
            timeout_seconds: 等待用户响应的超时时间
            auto_approve_on_timeout: 超时后是否自动批准
        """
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.auto_approve_on_timeout = auto_approve_on_timeout

        # 状态控制
        self.is_paused = False
        self.should_stop = False

        # 人工反馈记录
        self.feedback_history: List[Dict] = []

        # 实时配置覆盖
        self.config_overrides: Dict[str, Any] = {}

        cprint(f"🤝 人工介入管理器已启动 (模式: {mode.value})", "cyan")

    async def intervene(
        self,
        point: InterventionPoint,
        data: Dict[str, Any],
        allow_override: bool = True,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        人工介入点

        Args:
            point: 介入点类型
            data: 当前数据（决策、执行结果等）
            allow_override: 是否允许人工覆盖
            context: 额外上下文信息

        Returns:
            处理后的数据（可能被人工修改）
        """
        # 检查是否暂停
        await self._check_pause()

        # 检查是否应该停止
        if self.should_stop:
            raise KeyboardInterrupt("人工请求停止")

        # 根据模式决定是否需要介入
        if not self._should_intervene(point):
            return data

        # 显示当前状态
        self._display_intervention_info(point, data, context)

        # 根据模式处理
        if self.mode == InterventionMode.REVIEW_ONLY:
            # 仅展示，不阻塞
            await asyncio.sleep(2)  # 给用户时间阅读
            return data

        elif self.mode in [InterventionMode.APPROVE_TRADES, InterventionMode.APPROVE_ALL]:
            # 需要批准
            return await self._request_approval(point, data, allow_override, context)

        elif self.mode == InterventionMode.INTERACTIVE:
            # 完全交互模式
            return await self._interactive_session(point, data, context)

        return data

    def _should_intervene(self, point: InterventionPoint) -> bool:
        """判断是否需要介入"""
        if self.mode == InterventionMode.DISABLED:
            return False

        if self.mode == InterventionMode.REVIEW_ONLY:
            return True  # 所有点都展示，但不阻塞

        if self.mode == InterventionMode.APPROVE_TRADES:
            # 仅在交易执行前介入
            return point == InterventionPoint.AFTER_DECISION

        if self.mode == InterventionMode.APPROVE_ALL:
            # 所有关键点都介入
            return True

        if self.mode == InterventionMode.INTERACTIVE:
            return True

        return False

    def _display_intervention_info(
        self,
        point: InterventionPoint,
        data: Dict[str, Any],
        context: Optional[str],
    ) -> None:
        """显示介入信息"""
        cprint("\n" + "=" * 70, "yellow")
        cprint(f"🤝 人工介入点: {point.value}", "yellow", attrs=["bold"])
        cprint("=" * 70, "yellow")

        if context:
            cprint(f"\n📝 上下文: {context}", "cyan")

        # 根据介入点显示不同信息
        if point == InterventionPoint.AFTER_DECISION:
            self._display_decision(data)
        elif point == InterventionPoint.AFTER_EXECUTION:
            self._display_execution(data)
        elif point == InterventionPoint.BEFORE_EVALUATION:
            self._display_evaluation_context(data)

    def _display_decision(self, data: Dict[str, Any]) -> None:
        """显示决策信息"""
        decision = data.get('decision', {})
        market = data.get('market_snapshot')

        cprint("\n📊 LLM 决策:", "green", attrs=["bold"])
        cprint(f"  决策类型: {decision.get('decision_type', 'N/A')}", "white")

        signals = decision.get('signals', [])
        if signals:
            cprint(f"\n  信号 ({len(signals)} 个):", "white")
            for i, signal in enumerate(signals, 1):
                cprint(f"\n  [{i}] {signal.get('action', 'N/A')} {signal.get('symbol', '')}", "cyan")
                cprint(f"      数量: {signal.get('amount', 0)}", "white")
                cprint(f"      杠杆: {signal.get('leverage', 1)}x", "white")
                cprint(f"      止损: ${signal.get('stop_loss', 0):,.2f}", "white")
                cprint(f"      止盈: ${signal.get('take_profit', 0):,.2f}", "white")
                cprint(f"      信心: {signal.get('confidence', 0)}%", "white")
                cprint(f"      理由: {signal.get('reason', 'N/A')[:100]}", "white")

        # 显示分析摘要
        analysis = decision.get('analysis', '')
        if analysis:
            cprint(f"\n  分析摘要:\n  {analysis[:300]}...", "white")

    def _display_execution(self, data: Dict[str, Any]) -> None:
        """显示执行结果"""
        results = data.get('execution_results', [])

        cprint("\n📈 执行结果:", "green", attrs=["bold"])
        for i, r in enumerate(results, 1):
            signal = r.get('signal', {})
            result = r.get('result', {})
            success = "✅ 成功" if result.get('success') else "❌ 失败"
            cprint(f"\n  [{i}] {signal.get('action')} {signal.get('symbol')}: {success}", "cyan")
            if result.get('message'):
                cprint(f"      消息: {result.get('message')}", "white")

    def _display_evaluation_context(self, data: Dict[str, Any]) -> None:
        """显示评估上下文"""
        decision = data.get('decision', {})
        results = data.get('execution_results', [])

        cprint("\n🔍 评估准备:", "green", attrs=["bold"])
        cprint(f"  决策类型: {decision.get('decision_type', 'N/A')}", "white")
        cprint(f"  执行结果: {len(results)} 个", "white")

    async def _request_approval(
        self,
        point: InterventionPoint,
        data: Dict[str, Any],
        allow_override: bool,
        context: Optional[str],
    ) -> Dict[str, Any]:
        """请求人工批准"""
        cprint("\n" + "-" * 70, "yellow")
        cprint("请选择操作:", "yellow", attrs=["bold"])
        cprint("  [a] 批准 (Approve)", "green")
        cprint("  [r] 拒绝 (Reject)", "red")
        if allow_override:
            cprint("  [m] 修改 (Modify)", "cyan")
        cprint("  [p] 暂停 (Pause)", "yellow")
        cprint("  [s] 停止系统 (Stop)", "red")
        cprint("  [i] 查看详细信息 (Info)", "white")
        cprint("-" * 70, "yellow")

        try:
            # 使用 asyncio 实现超时
            choice = await asyncio.wait_for(
                asyncio.to_thread(input, "\n👉 请输入选择 (a/r/m/p/s/i): "),
                timeout=self.timeout_seconds
            )
            choice = choice.lower().strip()

        except asyncio.TimeoutError:
            if self.auto_approve_on_timeout:
                cprint(f"\n⏱️  超时 ({self.timeout_seconds}s)，自动批准", "yellow")
                return data
            else:
                cprint(f"\n⏱️  超时 ({self.timeout_seconds}s)，自动拒绝", "red")
                return self._reject_action(data)

        # 处理选择
        if choice == 'a':
            cprint("\n✅ 已批准", "green")
            self._record_feedback("approved", data, "")
            return data

        elif choice == 'r':
            cprint("\n❌ 已拒绝", "red")
            reason = input("拒绝原因 (可选): ")
            self._record_feedback("rejected", data, reason)
            return self._reject_action(data)

        elif choice == 'm' and allow_override:
            return await self._modify_data(data)

        elif choice == 'p':
            cprint("\n⏸️  系统已暂停", "yellow")
            self.is_paused = True
            await self._pause_menu()
            return await self._request_approval(point, data, allow_override, context)

        elif choice == 's':
            cprint("\n🛑 收到停止请求", "red")
            self.should_stop = True
            raise KeyboardInterrupt("人工请求停止")

        elif choice == 'i':
            self._display_detailed_info(data)
            return await self._request_approval(point, data, allow_override, context)

        else:
            cprint("\n❌ 无效选择，请重新输入", "red")
            return await self._request_approval(point, data, allow_override, context)

    async def _modify_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """修改数据"""
        cprint("\n✏️  修改模式", "cyan", attrs=["bold"])
        cprint("当前数据:", "white")
        print(json.dumps(data.get('decision', {}), indent=2, ensure_ascii=False))

        cprint("\n可修改的字段:", "yellow")
        cprint("  [1] 决策类型 (decision_type)", "white")
        cprint("  [2] 信号参数 (止损/止盈/数量等)", "white")
        cprint("  [3] 完全覆盖 (输入新的 JSON)", "white")
        cprint("  [0] 取消修改", "white")

        choice = input("\n请选择: ")

        if choice == '1':
            new_type = input("新的决策类型 (trade/hold/wait): ")
            if 'decision' in data:
                data['decision']['decision_type'] = new_type
                cprint(f"✅ 已修改决策类型为: {new_type}", "green")

        elif choice == '2':
            return await self._modify_signals(data)

        elif choice == '3':
            cprint("\n请输入新的决策 JSON:", "yellow")
            try:
                new_json = input()
                new_decision = json.loads(new_json)
                data['decision'] = new_decision
                cprint("✅ 已覆盖决策", "green")
            except json.JSONDecodeError as e:
                cprint(f"❌ JSON 解析失败: {e}", "red")

        self._record_feedback("modified", data, "人工修改")
        return data

    async def _modify_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """修改信号参数"""
        decision = data.get('decision', {})
        signals = decision.get('signals', [])

        if not signals:
            cprint("❌ 没有信号可修改", "red")
            return data

        cprint(f"\n选择要修改的信号 (1-{len(signals)}):", "yellow")
        for i, signal in enumerate(signals, 1):
            cprint(f"  [{i}] {signal.get('action')} {signal.get('symbol')}", "white")

        try:
            idx = int(input("\n信号编号: ")) - 1
            if 0 <= idx < len(signals):
                signal = signals[idx]

                cprint("\n当前参数:", "cyan")
                cprint(f"  止损: {signal.get('stop_loss')}", "white")
                cprint(f"  止盈: {signal.get('take_profit')}", "white")
                cprint(f"  数量: {signal.get('amount')}", "white")
                cprint(f"  杠杆: {signal.get('leverage')}", "white")

                # 修改参数
                new_sl = input(f"\n新止损 (回车跳过): ")
                if new_sl:
                    signal['stop_loss'] = float(new_sl)

                new_tp = input(f"新止盈 (回车跳过): ")
                if new_tp:
                    signal['take_profit'] = float(new_tp)

                new_amount = input(f"新数量 (回车跳过): ")
                if new_amount:
                    signal['amount'] = float(new_amount)

                cprint("\n✅ 信号已修改", "green")

        except (ValueError, IndexError) as e:
            cprint(f"❌ 输入错误: {e}", "red")

        return data

    async def _interactive_session(
        self,
        point: InterventionPoint,
        data: Dict[str, Any],
        context: Optional[str],
    ) -> Dict[str, Any]:
        """完全交互式会话"""
        while True:
            cprint("\n" + "-" * 70, "cyan")
            cprint("交互式模式 - 可用命令:", "cyan", attrs=["bold"])
            cprint("  [c] 继续 (Continue)", "green")
            cprint("  [m] 修改 (Modify)", "yellow")
            cprint("  [i] 详细信息 (Info)", "white")
            cprint("  [f] 提供反馈 (Feedback)", "cyan")
            cprint("  [p] 暂停 (Pause)", "yellow")
            cprint("  [q] 退出系统 (Quit)", "red")
            cprint("-" * 70, "cyan")

            choice = input("\n👉 命令: ").lower().strip()

            if choice == 'c':
                return data
            elif choice == 'm':
                data = await self._modify_data(data)
            elif choice == 'i':
                self._display_detailed_info(data)
            elif choice == 'f':
                await self._collect_feedback(data)
            elif choice == 'p':
                self.is_paused = True
                await self._pause_menu()
            elif choice == 'q':
                self.should_stop = True
                raise KeyboardInterrupt("人工请求停止")

    def _display_detailed_info(self, data: Dict[str, Any]) -> None:
        """显示详细信息"""
        cprint("\n" + "=" * 70, "cyan")
        cprint("详细信息", "cyan", attrs=["bold"])
        cprint("=" * 70, "cyan")
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        cprint("=" * 70, "cyan")

    async def _collect_feedback(self, data: Dict[str, Any]) -> None:
        """收集人工反馈"""
        cprint("\n📝 请提供反馈:", "cyan")
        feedback = input("反馈内容: ")

        if feedback:
            self._record_feedback("feedback", data, feedback)
            cprint("✅ 反馈已记录，将用于改进系统", "green")

    def _reject_action(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """拒绝操作"""
        # 将决策类型改为 wait
        if 'decision' in data:
            data['decision']['decision_type'] = 'wait'
            data['decision']['signals'] = []
            data['decision']['human_rejected'] = True
        return data

    def _record_feedback(
        self,
        action: str,
        data: Dict[str, Any],
        comment: str,
    ) -> None:
        """记录人工反馈"""
        self.feedback_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'data_summary': self._summarize_data(data),
            'comment': comment,
        })

    def _summarize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """数据摘要（避免存储过多信息）"""
        decision = data.get('decision', {})
        return {
            'decision_type': decision.get('decision_type'),
            'signals_count': len(decision.get('signals', [])),
        }

    async def _check_pause(self) -> None:
        """检查暂停状态"""
        while self.is_paused:
            await asyncio.sleep(0.5)

    async def _pause_menu(self) -> None:
        """暂停菜单"""
        while self.is_paused:
            cprint("\n" + "-" * 70, "yellow")
            cprint("⏸️  系统已暂停", "yellow", attrs=["bold"])
            cprint("  [r] 恢复 (Resume)", "green")
            cprint("  [s] 停止系统 (Stop)", "red")
            cprint("  [c] 查看配置 (Config)", "cyan")
            cprint("-" * 70, "yellow")

            choice = input("\n👉 选择: ").lower().strip()

            if choice == 'r':
                self.is_paused = False
                cprint("\n▶️  系统已恢复", "green")
                break
            elif choice == 's':
                self.should_stop = True
                raise KeyboardInterrupt("人工请求停止")
            elif choice == 'c':
                self._display_config()

    def _display_config(self) -> None:
        """显示当前配置"""
        cprint("\n⚙️  当前配置:", "cyan")
        cprint(f"  介入模式: {self.mode.value}", "white")
        cprint(f"  超时时间: {self.timeout_seconds}s", "white")
        cprint(f"  超时自动批准: {self.auto_approve_on_timeout}", "white")
        cprint(f"  反馈记录数: {len(self.feedback_history)}", "white")

    def get_feedback_summary(self) -> Dict[str, Any]:
        """获取反馈摘要"""
        if not self.feedback_history:
            return {"total": 0, "actions": {}}

        actions = {}
        for fb in self.feedback_history:
            action = fb['action']
            actions[action] = actions.get(action, 0) + 1

        return {
            'total': len(self.feedback_history),
            'actions': actions,
            'recent': self.feedback_history[-5:],
        }

    def export_feedback(self, file_path: str) -> None:
        """导出反馈记录"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_history, f, indent=2, ensure_ascii=False)
            cprint(f"✅ 反馈已导出到: {file_path}", "green")
        except Exception as e:
            cprint(f"❌ 导出失败: {e}", "red")