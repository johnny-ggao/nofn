"""
生成记忆摘要的工具脚本

用法:
    uv run python generate_summary.py
"""
import asyncio
from pathlib import Path

async def main():
    from src.learning import MemoryManager
    from langchain_openai import ChatOpenAI
    from src.utils.config import config
    from termcolor import cprint

    # 加载配置
    llm_config = config.llm

    # 初始化 LLM
    llm = ChatOpenAI(
        model=llm_config.model,
        temperature=llm_config.temperature,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
    )

    # 初始化 MemoryManager
    memory = MemoryManager(storage_dir="data/memory", llm=llm)

    cprint("\n" + "=" * 70, "cyan")
    cprint("📝 记忆摘要生成器", "cyan")
    cprint("=" * 70, "cyan")

    cprint(f"\n📊 当前状态:", "yellow")
    cprint(f"  - 详细案例: {len(memory.cases)} 个", "white")
    cprint(f"  - 历史摘要: {len(memory.summaries)} 个", "white")

    # 生成每周摘要
    cprint("\n🔄 开始生成每周摘要...", "cyan")
    summary = await memory.generate_weekly_summary()

    if summary:
        cprint("\n✅ 摘要生成成功!", "green")
        cprint(f"\n摘要 ID: {summary.summary_id}", "white")
        cprint(f"时间范围: {summary.period_start.strftime('%Y-%m-%d')} - {summary.period_end.strftime('%Y-%m-%d')}", "white")
        cprint(f"案例数: {summary.total_cases}, 交易数: {summary.total_trades}", "white")
        cprint(f"胜率: {summary.win_rate*100:.1f}%, 夏普: {summary.sharpe_ratio:.2f}", "white")

        if summary.key_patterns:
            cprint("\n📌 关键模式:", "yellow")
            for i, pattern in enumerate(summary.key_patterns, 1):
                cprint(f"  {i}. {pattern}", "white")

        if summary.successful_strategies:
            cprint("\n✅ 成功策略:", "green")
            for i, strategy in enumerate(summary.successful_strategies, 1):
                cprint(f"  {i}. {strategy}", "white")

        if summary.failed_strategies:
            cprint("\n❌ 失败策略:", "red")
            for i, strategy in enumerate(summary.failed_strategies, 1):
                cprint(f"  {i}. {strategy}", "white")

        if summary.lessons:
            cprint("\n💡 核心经验:", "magenta")
            for i, lesson in enumerate(summary.lessons, 1):
                cprint(f"  {i}. {lesson}", "white")

        if summary.market_insights:
            cprint("\n🌍 市场洞察:", "cyan")
            cprint(f"  {summary.market_insights}", "white")

        cprint(f"\n💾 摘要已保存到: data/memory/summaries.json", "green")
        cprint(f"📊 清理后剩余案例: {len(memory.cases)} 个", "white")
    else:
        cprint("\n⚠️  案例不足或生成失败，跳过摘要", "yellow")
        cprint("  提示: 至少需要5个案例才能生成摘要", "white")

    cprint("\n" + "=" * 70, "cyan")

if __name__ == "__main__":
    asyncio.run(main())
