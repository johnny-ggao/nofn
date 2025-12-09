"""决策历史摘要生成器。

使用 LLM 将历史决策记忆压缩成简洁的摘要。
"""

from typing import List, Optional

from termcolor import cprint

from .state import (
    DecisionMemory,
    DecisionSummary,
    create_summary_from_memories,
    format_memories_for_summarization,
)


SUMMARIZE_PROMPT_TEMPLATE = """你是一个交易决策分析助手。请分析以下交易决策历史，生成一个简洁的摘要。

## 决策历史
{memories_text}

## 要求
请用 2-3 句话总结：
1. 主要的交易行为模式（开多/开空/平仓的频率和原因）
2. 执行效果（成功率、盈亏情况）
3. 需要注意的教训或趋势

只输出摘要内容，不要添加其他格式或前缀。"""


async def generate_summary_with_llm(
    memories: List[DecisionMemory],
    llm_call: Optional[callable] = None,
) -> str:
    """使用 LLM 生成决策历史摘要。

    Args:
        memories: 要摘要的决策记忆列表
        llm_call: 可选的 LLM 调用函数，签名为 async (prompt: str) -> str
                  如果不提供，使用默认的简单摘要

    Returns:
        摘要文本
    """
    if not memories:
        return "无历史决策记录。"

    # 格式化记忆
    memories_text = format_memories_for_summarization(memories)

    # 如果提供了 LLM 调用函数，使用它
    if llm_call:
        try:
            prompt = SUMMARIZE_PROMPT_TEMPLATE.format(memories_text=memories_text)
            summary = await llm_call(prompt)
            return summary.strip()
        except Exception as e:
            cprint(f"LLM 摘要生成失败，使用默认摘要: {e}", "yellow")

    # 默认摘要（不依赖 LLM）
    return _generate_simple_summary(memories)


def _generate_simple_summary(memories: List[DecisionMemory]) -> str:
    """生成简单的统计摘要（不依赖 LLM）。

    当 LLM 不可用时使用。
    """
    if not memories:
        return "无历史决策记录。"

    # 统计
    total = len(memories)
    executed = sum(1 for m in memories if m["executed"])
    total_pnl = sum(m.get("realized_pnl") or 0.0 for m in memories)

    # 行为分类
    actions = {}
    for m in memories:
        action = m.get("action", "unknown")
        actions[action] = actions.get(action, 0) + 1

    # 构建摘要
    action_desc = ", ".join(f"{k}={v}次" for k, v in actions.items())
    pnl_desc = f"盈利{total_pnl:.4f}" if total_pnl >= 0 else f"亏损{abs(total_pnl):.4f}"

    cycle_start = memories[0]["cycle_index"]
    cycle_end = memories[-1]["cycle_index"]

    return (
        f"周期{cycle_start}-{cycle_end}共{total}次决策，执行{executed}次。"
        f"行为分布：{action_desc}。累计{pnl_desc}。"
    )


async def maybe_generate_summary(
    memories: List[DecisionMemory],
    llm_call: Optional[callable] = None,
    threshold: int = 10,
) -> Optional[DecisionSummary]:
    """判断是否需要生成摘要，如果需要则生成。

    当 memories 达到 threshold 时，将前半部分压缩成摘要。

    Args:
        memories: 当前的决策记忆列表
        llm_call: LLM 调用函数
        threshold: 触发摘要的阈值（默认 10）

    Returns:
        如果生成了摘要返回 DecisionSummary，否则返回 None
    """
    if len(memories) < threshold:
        return None

    # 取前半部分用于摘要
    half = threshold // 2
    to_summarize = memories[:half]

    # 生成摘要内容
    summary_content = await generate_summary_with_llm(to_summarize, llm_call)

    # 创建摘要对象
    summary = create_summary_from_memories(to_summarize, summary_content)

    cprint(
        f"生成决策摘要: 周期{summary['cycle_range'][0]}-{summary['cycle_range'][1]}, "
        f"{summary['total_decisions']}次决策, PnL={summary['total_pnl']:.4f}",
        "cyan"
    )

    return summary
