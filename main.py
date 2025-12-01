"""
NOFN 自主交易系统 - 主程序
"""
import asyncio
from dotenv import load_dotenv
from termcolor import cprint

from src.engine.trading_engine import TradingEngine
from src.adapters.factory import AdapterFactory
from src.learning.graph import TradingWorkflowGraph
from src.utils import config


async def main():
    """主函数"""
    load_dotenv()

    exchange = config.strategy.exchange
    symbols = config.strategy.symbols
    interval_seconds = config.strategy.interval_seconds

    # 交易所配置
    exchange_config = config.get_exchange_config(exchange)

    cprint("\n" + "=" * 70, "cyan", attrs=["bold"])
    cprint("🚀 NOFN 交易系统 (LangGraph)", "cyan", attrs=["bold"])
    cprint("=" * 70, "cyan", attrs=["bold"])
    cprint(f"\n📊 交易所: {exchange.upper()}", "white")
    cprint(f"💰 监控币种: {', '.join(symbols)}", "white")
    cprint(f"⏱️  循环间隔: {interval_seconds}秒 ({interval_seconds / 60:.1f}分钟)", "white")
    cprint(f"🤖 LLM: {config.llm.provider}/{config.llm.model}", "white")
    cprint("")

    try:
        # 1. 创建交易所适配器
        cprint("🔧 初始化交易引擎...", "cyan")
        cprint(f"   支持的交易所: {', '.join(AdapterFactory.list_available())}", "white")
        cprint(f"   当前配置: {exchange}", "white")

        adapter = await AdapterFactory.create(
            name=exchange,
            api_key=exchange_config.api_key,
            api_secret=exchange_config.api_secret,
            testnet=exchange_config.testnet if hasattr(exchange_config, 'testnet') else False,
        )

        # 2. 创建交易引擎
        engine = TradingEngine(adapter=adapter)
        cprint("✅ 交易引擎初始化完成", "green")

        # 3. 创建工作流图
        cprint("\n📊 创建工作流图...", "cyan")
        workflow = TradingWorkflowGraph(
            engine=engine,
            llm_config=config.llm,
        )

        # 4. 运行交易循环
        cprint("\n" + "=" * 70, "green", attrs=["bold"])
        cprint("▶️  开始交易循环", "green", attrs=["bold"])
        cprint("=" * 70, "green", attrs=["bold"])
        cprint("")

        iteration = 0

        while True:
            try:
                final_state = await workflow.run_iteration(
                    symbols=symbols,
                    iteration=iteration,
                )

                iteration += 1

                # 显示状态摘要
                cprint("\n" + "─" * 70, "white")
                cprint("📊 本次迭代摘要:", "white", attrs=["bold"])
                cprint(f"  决策类型: {final_state.get('decision', {}).get('decision_type', 'N/A')}", "white")
                cprint(f"  执行结果: {len(final_state.get('execution_results', []))} 个", "white")
                cprint(f"  质量评分: {final_state.get('quality_score', 'N/A')}", "white")
                cprint(f"  学到的经验: {len(final_state.get('lessons_learned', []))} 条", "white")
                cprint("─" * 70, "white")

                cprint(f"\n⏳ 等待 {interval_seconds} 秒进入下一轮...\n", "cyan")
                await asyncio.sleep(interval_seconds)

            except KeyboardInterrupt:
                cprint("\n⚠️  收到中断信号，正在停止...", "yellow")
                break
            except Exception as e:
                cprint(f"\n❌ 迭代 {iteration} 执行失败: {e}", "red")
                import traceback
                traceback.print_exc()

                await asyncio.sleep(60)

    except Exception as e:
        cprint(f"\n❌ 系统初始化失败: {e}", "red")
        import traceback
        traceback.print_exc()

    finally:
        if 'adapter' in locals():
            cprint("\n🔧 关闭适配器...", "cyan")
            await adapter.close()

        cprint("\n👋 交易系统已停止\n", "yellow")


if __name__ == "__main__":
    asyncio.run(main())