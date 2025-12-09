"""
NOFN Trading System - ValueCell Style Architecture

Main entry point for the trading system.
"""

import argparse
import asyncio
import sys

from dotenv import load_dotenv
from termcolor import cprint

from src.config import get_settings, load_dotenv as load_env
from src.trading import (
    ExchangeConfig,
    LLMModelConfig,
    MarginMode,
    MarketType,
    SummaryLLMConfig,
    TradingConfig,
    TradingMode,
    UserRequest,
)
from src.strategy import StrategyAgent


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="NOFN Trading System")
    parser.add_argument(
        "--template", "-t",
        type=str,
        default=None,
        help="Strategy template: default, aggressive, insane, funding_rate (or path to custom template)",
    )
    parser.add_argument(
        "--symbols", "-s",
        type=str,
        nargs="+",
        default=None,
        help="Trading symbols (e.g., BTC/USDT:USDT ETH/USDT:USDT)",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["live", "virtual"],
        default=None,
        help="Trading mode: live or virtual",
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=None,
        help="Decision interval in seconds",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available templates and exit",
    )
    parser.add_argument(
        "--reflection", "-r",
        action="store_true",
        help="Enable reflection mode (自动分析历史表现并调整策略)",
    )
    return parser.parse_args()


def create_user_request(args) -> UserRequest:
    """Create user request from settings and command line args."""
    settings = get_settings()

    # Build summary LLM config if configured
    summary_llm_config = None
    if settings.llm.summary and settings.llm.summary.enabled:
        summary_llm_config = SummaryLLMConfig(
            enabled=True,
            provider=settings.llm.summary.provider,
            model_id=settings.llm.summary.model,
            api_key=settings.secrets.get_llm_api_key(settings.llm.summary.provider),
            base_url=settings.llm.summary.base_url,
            temperature=settings.llm.summary.temperature,
        )
        cprint(
            f"Summary LLM 已配置: {settings.llm.summary.provider}/{settings.llm.summary.model}",
            "white"
        )

    # Build LLM config from YAML + secrets
    llm_config = LLMModelConfig(
        provider=settings.llm.provider,
        model_id=settings.llm.model,
        api_key=settings.get_llm_api_key(),
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        summary_llm=summary_llm_config,
    )

    # Build exchange config from YAML + secrets
    trading_mode = (
        TradingMode.LIVE
        if settings.strategy.trading_mode.lower() == "live"
        else TradingMode.VIRTUAL
    )

    market_type_map = {
        "spot": MarketType.SPOT,
        "future": MarketType.FUTURE,
        "swap": MarketType.SWAP,
    }
    market_type = market_type_map.get(
        settings.exchange.market_type.lower(),
        MarketType.SWAP,
    )

    margin_mode_map = {
        "isolated": MarginMode.ISOLATED,
        "cross": MarginMode.CROSS,
    }
    margin_mode = margin_mode_map.get(
        settings.exchange.margin_mode.lower(),
        MarginMode.CROSS,
    )

    # Get settle_coin from config (default to USDT)
    settle_coin = getattr(settings.exchange, "settle_coin", "USDT")

    exchange_config = ExchangeConfig(
        exchange_id=settings.exchange.id,
        trading_mode=trading_mode,
        api_key=settings.get_exchange_api_key(),
        secret_key=settings.get_exchange_secret_key(),
        passphrase=settings.get_exchange_passphrase(),
        testnet=settings.exchange.testnet,
        market_type=market_type,
        margin_mode=margin_mode,
        settle_coin=settle_coin,
    )

    # Command line args override config file
    template = args.template if args.template else settings.strategy.template
    symbols = args.symbols if args.symbols else settings.strategy.symbols
    decide_interval = args.interval if args.interval else settings.strategy.decide_interval

    # Build trading config from YAML + command line overrides
    trading_config = TradingConfig(
        strategy_name=settings.strategy.name,
        template_id=template,
        symbols=symbols,
        initial_capital=settings.strategy.initial_capital,
        max_leverage=settings.strategy.max_leverage,
        max_positions=settings.strategy.max_positions,
        decide_interval=decide_interval,
    )

    return UserRequest(
        llm_model_config=llm_config,
        exchange_config=exchange_config,
        trading_config=trading_config,
    )


async def main():
    """Main function."""
    args = parse_args()

    if args.list_templates:
        from src.strategy import list_templates
        cprint("可用模板:", "white")
        for name in list_templates():
            cprint(f"  - {name}", "white")
        return

    # 记载环境变量
    load_dotenv()
    load_env()

    # 读取配置文件
    settings = get_settings()

    template = args.template if args.template else settings.strategy.template
    symbols = args.symbols if args.symbols else settings.strategy.symbols
    mode = args.mode if args.mode else settings.strategy.trading_mode
    interval = args.interval if args.interval else settings.strategy.decide_interval

    cprint("=" * 50, "cyan")
    cprint("NOFN Trading System", "cyan", attrs=["bold"])
    cprint("=" * 50, "cyan")

    # 反思模式
    enable_reflection = args.reflection

    cprint(f"交易所: {settings.exchange.id.upper()}", "white")
    cprint(f"交易对: {symbols}", "white")
    cprint(f"策略模版: {template}", "white")
    cprint(f"运行间隔: {interval}s", "white")
    cprint(f"模式: {mode.upper()}", "white")
    cprint(f"LLM: {settings.llm.provider}/{settings.llm.model}", "white")

    agent = None
    try:
        request = create_user_request(args)

        cprint("正在初始化策略代理...", "white")
        agent = StrategyAgent(
            request,
            enable_reflection=enable_reflection,
        )

        strategy_id = await agent.start()
        cprint(f"✓ 策略已启动: {strategy_id}", "green")

        cprint("=" * 50, "cyan")
        cprint("开始交易循环", "cyan", attrs=["bold"])
        cprint("=" * 50, "cyan")

        # Run the agent
        await agent.run()

    except KeyboardInterrupt:
        cprint("收到中断信号，正在停止...", "yellow")
        if agent:
            await agent.stop()
    except Exception as e:
        import traceback
        cprint(f"✗ 系统错误: {e}", "red")
        traceback.print_exc()
    finally:
        cprint("交易系统已停止", "yellow")


if __name__ == "__main__":
    asyncio.run(main())
