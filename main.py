#!/usr/bin/env python3
"""
NoFn Trading System - Main Entry Point

完全基于 Agno 的三层架构:
- Layer 1: TradingEngine (执行层) - 快速、确定性
- Layer 2: TradingAgents (决策层) - Agno Agent 集合
- Layer 3: LearningGraph (学习层) - Agno 原生记忆和工作流
"""
import asyncio
import sys
from pathlib import Path

from termcolor import cprint

from src.utils.config import config

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


async def run_trading_system():
    """运行基于 Agno 的交易系统"""
    # 读取配置
    strategy_config = config.strategy
    llm_config = config.get_llm_config()
    exchange = strategy_config.exchange
    symbols = strategy_config.symbols

    cprint("=" * 70, "cyan")
    cprint("🤖 NoFn Trading System", "cyan")
    cprint("=" * 70, "cyan")
    cprint(f"交易所: {exchange}", "cyan")
    cprint(f"交易对: {', '.join(symbols)}", "cyan")
    cprint(f"循环间隔: {strategy_config.interval_seconds}s ({strategy_config.interval_seconds / 60:.1f} 分钟)", "cyan")
    cprint(f"LLM: {llm_config.provider} ({llm_config.model})", "cyan")
    cprint("=" * 70, "cyan")

    adapter = None

    try:
        # Layer 1: 执行层
        from src.adapters import HyperliquidAdapter, BinanceAdapter
        from src.engine import TradingEngine
        from src.models import TradeHistoryManager

        exchange_config = config.get_exchange_config(exchange)

        # 根据配置的交易所创建对应的适配器
        if exchange.lower() == "hyperliquid":
            adapter = HyperliquidAdapter(
                api_key=exchange_config.api_key,
                api_secret=exchange_config.api_secret,
                testnet=exchange_config.testnet,
            )
        elif exchange.lower() in ["binance", "binance_usdt"]:
            adapter = BinanceAdapter(
                api_key=exchange_config.api_key,
                api_secret=exchange_config.api_secret,
                margin_type="USDT",
                testnet=exchange_config.testnet,
            )
        elif exchange.lower() == "binance_usdc":
            adapter = BinanceAdapter(
                api_key=exchange_config.api_key,
                api_secret=exchange_config.api_secret,
                margin_type="USDC",
                testnet=exchange_config.testnet,
            )
        else:
            raise ValueError(f"不支持的交易所: {exchange}，支持: hyperliquid, binance, binance_usdt, binance_usdc")

        await adapter.initialize()

        trade_history = TradeHistoryManager()
        engine = TradingEngine(adapter=adapter, trade_history=trade_history)
        cprint("✅ Layer 1 初始化完成", "green")

        # Layer 2 & 3: 决策层和学习层
        from src.learning import LearningGraph

        # 获取系统提示词路径
        system_prompt_path = str(Path(__file__).parent / "src" / "prompts" / "nofn_v2.txt")

        learning_graph = LearningGraph(
            engine=engine,
            db_path="data/agno_trading.db",
            model_provider=llm_config.provider,
            model_id=llm_config.model,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            temperature=llm_config.temperature,
            system_prompt_path=system_prompt_path,
        )
        cprint("✅ Layer 2 & 3 (决策层 + 学习层) 初始化完成", "green")

        cprint("\n" + "🚀" * 35, "green")
        cprint("🚀 所有组件初始化完成，开始交易循环！", "green")
        cprint("🚀" * 35 + "\n", "green")

        # 运行循环
        await learning_graph.run_loop(
            symbols=symbols,
            interval_seconds=strategy_config.interval_seconds,
            max_iterations=strategy_config.max_iterations,
        )

    except KeyboardInterrupt:
        cprint("=" * 70, "yellow")
        cprint("⚠️  收到中断信号，正在停止...", "yellow")
        cprint("=" * 70, "yellow")

    except Exception as e:
        cprint(f"❌ 系统运行出错: {e}", "red")
        import traceback
        traceback.print_exc()

    finally:
        if adapter:
            try:
                await adapter.close()
            except Exception:
                pass
        cprint("👋 系统已停止", "yellow")


async def main():
    """主入口"""
    await run_trading_system()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        cprint(f"❌ 启动失败: {e}", "red")
        sys.exit(1)
