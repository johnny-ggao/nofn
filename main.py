#!/usr/bin/env python3
"""
NoFn Trading System - Main Entry Point

Simple and direct entry point for running the trading system.
"""
import asyncio
import sys
from pathlib import Path

from termcolor import cprint

from src.utils.config import config

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

async def run_strategy_mode():
    """运行 Hansen 自主学习交易智能体"""
    # 从配置读取
    strategy_config = config.strategy
    llm_config = config.get_llm_config()

    exchange = strategy_config.exchange
    symbols = strategy_config.symbols

    cprint("=" * 70, "cyan")
    cprint("🤖 NoFn Trading Agent - 自主学习量化交易智能体", "cyan")
    cprint("=" * 70, "cyan")
    cprint(f"交易所: {exchange}", "cyan")
    cprint(f"交易对: {', '.join(symbols)}", "cyan")
    cprint(f"循环间隔: {strategy_config.interval_seconds}s ({strategy_config.interval_seconds / 60:.1f} 分钟)", "cyan")
    cprint(f"最大迭代: {strategy_config.max_iterations or '无限'}", "cyan")
    cprint(f"LLM: {llm_config.provider} ({llm_config.model})", "cyan")
    cprint("=" * 70, "cyan")

    from src.adapters import HyperliquidAdapter

    exchange_config = config.get_exchange_config(exchange)

    adapter = HyperliquidAdapter(
        api_key=exchange_config.api_key,
        api_secret=exchange_config.api_secret,
        testnet=exchange_config.testnet,
    )

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=llm_config.model,
            temperature=llm_config.temperature,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
        )

        from src.agents.hansen.trading_agent import TradingAgent

        agent = TradingAgent(
            adapter=adapter,
            llm=llm,
            config={
                "symbols": symbols,
            }
        )

        cprint(f"🚀 Tool-based Agent 已启动，开始交易循环...", "green")

        await agent.run_loop(
            interval_seconds=strategy_config.interval_seconds,
            max_iterations=strategy_config.max_iterations,
        )

    except KeyboardInterrupt:
        cprint("=" * 70, "yellow")
        cprint("⚠️ 收到中断信号，正在停止...", "yellow")
        cprint("=" * 70, "yellow")

    except Exception as e:
        cprint(f"❌ Agent 运行时出错: {e}", "red")

    finally:
        cprint("👋 Agent 已停止", "yellow")


async def main():
    """主入口"""
    await run_strategy_mode()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        cprint(f"❌ 启动失败: {e}", "red")
        sys.exit(1)
