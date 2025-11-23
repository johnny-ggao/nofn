"""
ACE LangGraph Nodes

将 Generator、Reflector、Curator 包装为 LangGraph 节点函数
"""

from termcolor import cprint
from .state import ACEState
from ..core import Generator, Reflector, Curator
from ..storage import ContextStore


def create_generator_node(generator: Generator, context_store: ContextStore):
    """创建 Generator 节点"""

    async def generator_node(state: ACEState) -> ACEState:
        """
        Generator Phase: 获取市场数据、检索知识、LLM决策、执行交易

        Returns:
            更新后的状态，包含 trace
        """
        try:
            cprint("\n" + "┌" + "─" * 68 + "┐", "cyan")
            cprint("│" + " " * 24 + "Generator Phase" + " " * 29 + "│", "cyan", attrs=['bold'])
            cprint("└" + "─" * 68 + "┘", "cyan")

            # 执行 Generator
            trace = await generator.execute(state["symbols"])

            # 保存到数据库
            context_store.save_trace(trace)

            return {
                "trace": trace,
                "errors": state.get("errors", [])
            }

        except Exception as e:
            cprint(f"❌ Generator 节点失败: {e}", "red")
            errors = state.get("errors", [])
            errors.append(f"Generator: {str(e)}")
            import traceback
            traceback.print_exc()

            return {
                "trace": None,
                "errors": errors
            }

    return generator_node


def create_reflector_node(reflector: Reflector, context_store: ContextStore):
    """创建 Reflector 节点"""

    async def reflector_node(state: ACEState) -> ACEState:
        """
        Reflector Phase: 反思执行结果，评估策略有效性

        Returns:
            更新后的状态，包含 reflection
        """
        try:
            cprint("\n" + "┌" + "─" * 68 + "┐", "magenta")
            cprint("│" + " " * 24 + "Reflector Phase" + " " * 28 + "│", "magenta", attrs=['bold'])
            cprint("└" + "─" * 68 + "┘", "magenta")

            trace = state.get("trace")
            if not trace:
                cprint("⚠️  无 trace，跳过 Reflector", "yellow")
                return {"reflection": None, "errors": state.get("errors", [])}

            # 执行 Reflector
            reflection = await reflector.reflect(trace)

            # 保存到数据库
            context_store.save_reflection(reflection)

            return {
                "reflection": reflection,
                "errors": state.get("errors", [])
            }

        except Exception as e:
            cprint(f"❌ Reflector 节点失败: {e}", "red")
            errors = state.get("errors", [])
            errors.append(f"Reflector: {str(e)}")
            import traceback
            traceback.print_exc()

            return {
                "reflection": None,
                "errors": errors
            }

    return reflector_node


def create_curator_node(curator: Curator):
    """创建 Curator 节点"""

    async def curator_node(state: ACEState) -> ACEState:
        """
        Curator Phase: 根据反思更新知识库

        Returns:
            更新后的状态，包含 updated_entries
        """
        try:
            cprint("\n" + "┌" + "─" * 68 + "┐", "blue")
            cprint("│" + " " * 25 + "Curator Phase" + " " * 30 + "│", "blue", attrs=['bold'])
            cprint("└" + "─" * 68 + "┘", "blue")

            reflection = state.get("reflection")
            if not reflection:
                cprint("⚠️  无 reflection，跳过 Curator", "yellow")
                return {"updated_entries": [], "errors": state.get("errors", [])}

            # 执行 Curator
            updated_entries = await curator.curate(reflection)

            return {
                "updated_entries": updated_entries,
                "errors": state.get("errors", [])
            }

        except Exception as e:
            cprint(f"❌ Curator 节点失败: {e}", "red")
            errors = state.get("errors", [])
            errors.append(f"Curator: {str(e)}")
            import traceback
            traceback.print_exc()

            return {
                "updated_entries": [],
                "errors": errors
            }

    return curator_node


def create_summary_node():
    """创建总结节点 - 打印本次迭代摘要"""

    def summary_node(state: ACEState) -> ACEState:
        """
        Summary: 打印本次迭代的摘要信息
        """
        try:
            trace = state.get("trace")
            reflection = state.get("reflection")
            updated_entries = state.get("updated_entries", [])

            cprint("\n" + "┌" + "─" * 68 + "┐", "white")
            cprint("│" + " " * 28 + "迭代摘要" + " " * 32 + "│", "white", attrs=['bold'])
            cprint("└" + "─" * 68 + "┘", "white")

            if trace and trace.decisions:
                cprint(f"\n📊  决策数量: {len(trace.decisions)}", "white", attrs=['bold'])

                for i, decision in enumerate(trace.decisions, 1):
                    action = decision.action
                    symbol = decision.symbol
                    conf = decision.confidence

                    action_emoji = {
                        'open_long': '📈',
                        'open_short': '📉',
                        'close': '✖️',
                        'hold': '⏸️',
                        'adjust': '🔧'
                    }.get(action, '❓')

                    cprint(f"\n{i}. {action_emoji}  {symbol}: {action.upper()}", "white", attrs=['bold'])
                    cprint(f"   置信度: {conf:.2%}", "white")

                    if decision.reasoning:
                        reasoning_preview = decision.reasoning[:80]
                        cprint(f"   推理: {reasoning_preview}...", "white")

            # 执行结果
            if trace:
                if trace.execution_success:
                    cprint(f"\n✅  执行: 成功", "green", attrs=['bold'])
                else:
                    cprint(f"\n❌  执行: 失败", "red", attrs=['bold'])
                    if trace.execution_errors:
                        for error in trace.execution_errors:
                            cprint(f"   错误: {error}", "red")

                # 盈亏
                if trace.account_change and trace.account_change.get('pnl') is not None:
                    pnl_value = float(trace.account_change['pnl'])
                    if pnl_value > 0:
                        cprint(f"💰  盈亏: +${pnl_value:.2f}", "green", attrs=['bold'])
                    elif pnl_value < 0:
                        cprint(f"💸  盈亏: -${abs(pnl_value):.2f}", "red", attrs=['bold'])
                    else:
                        cprint(f"➖  盈亏: $0.00", "white")

            # 反思结果
            if reflection:
                cprint(f"\n🤔  反思结果:", "magenta", attrs=['bold'])
                cprint(f"   成功: {'是' if reflection.is_successful else '否'}", "white")
                if reflection.failure_type.value != 'none':
                    cprint(f"   失败类型: {reflection.failure_type.value}", "yellow")

                # 新洞察
                if reflection.key_insights:
                    cprint(f"\n💡  新洞察 ({len(reflection.key_insights)} 条):", "cyan", attrs=['bold'])
                    for i, insight in enumerate(reflection.key_insights[:3], 1):
                        cprint(f"   {i}. {insight}", "white")
                    if len(reflection.key_insights) > 3:
                        cprint(f"   ... 还有 {len(reflection.key_insights) - 3} 条", "white")

            # Curator 结果
            cprint(f"\n📝  Curator:", "blue", attrs=['bold'])
            cprint(f"   更新/创建: {len(updated_entries)} 个条目", "white")

            cprint("\n" + "─" * 70, "white")

            # 返回状态不变
            return {}

        except Exception as e:
            cprint(f"❌ Summary 节点失败: {e}", "red")
            return {}

    return summary_node


def create_maintenance_node(context_store: ContextStore):
    """创建维护节点 - 周期性知识库清理"""

    async def maintenance_node(state: ACEState) -> ACEState:
        """
        Maintenance: 周期性维护知识库
        """
        try:
            iteration = state.get("iteration", 0)

            # 每 10 次迭代执行一次维护
            if iteration % 10 != 0:
                return {}

            cprint("\n" + "┌" + "─" * 68 + "┐", "yellow")
            cprint("│" + " " * 25 + "知识库维护" + " " * 31 + "│", "yellow", attrs=['bold'])
            cprint("└" + "─" * 68 + "┘", "yellow")

            # 清理低置信度条目
            deleted = context_store.prune_low_confidence_entries(threshold=0.2)
            if deleted > 0:
                cprint(f"🗑️  清理了 {deleted} 个低置信度条目 (< 0.2)", "yellow")

            # 归档旧条目
            archived = context_store.archive_old_entries(days=90)
            if archived > 0:
                cprint(f"📦  归档了 {archived} 个旧条目 (> 90 天未使用)", "yellow")

            if deleted == 0 and archived == 0:
                cprint("✅  无需清理", "white")

            return {}

        except Exception as e:
            cprint(f"❌ Maintenance 节点失败: {e}", "red")
            return {}

    return maintenance_node
