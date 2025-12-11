"""
NOFN Trading System - ValueCell Style Architecture

Main entry point for the trading system.
"""

import asyncio

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


def create_user_request() -> UserRequest:
    """根据配置文件创建用户请求"""
    settings = get_settings()

    # 如果已配置，则构建 LLM 配置摘要
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

    # 从 YAML 和密钥构建 LLM 配置
    llm_config = LLMModelConfig(
        provider=settings.llm.provider,
        model_id=settings.llm.model,
        api_key=settings.get_llm_api_key(),
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        summary_llm=summary_llm_config,
    )

    # 从 YAML 和密钥构建交易所配置
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

    # 从配置中获取结算币（默认为 USDT）
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

    # 从配置文件构建交易配置
    trading_config = TradingConfig(
        strategy_name=settings.strategy.name,
        template_id=settings.strategy.template,
        symbols=settings.strategy.symbols,
        initial_capital=settings.strategy.initial_capital,
        max_leverage=settings.strategy.max_leverage,
        max_positions=settings.strategy.max_positions,
        decide_interval=settings.strategy.decide_interval,
    )

    return UserRequest(
        llm_model_config=llm_config,
        exchange_config=exchange_config,
        trading_config=trading_config,
    )


async def main():
    """Main function."""
    # 加载环境变量
    load_dotenv()
    load_env()

    # 读取配置文件
    settings = get_settings()

    cprint("=" * 50, "cyan")
    cprint("NOFN Trading System", "cyan", attrs=["bold"])
    cprint("=" * 50, "cyan")

    cprint(f"交易所: {settings.exchange.id.upper()}", "white")
    cprint(f"交易对: {settings.strategy.symbols}", "white")
    cprint(f"策略模版: {settings.strategy.template}", "white")
    cprint(f"运行间隔: {settings.strategy.decide_interval}s", "white")
    cprint(f"模式: {settings.strategy.trading_mode.upper()}", "white")
    cprint(f"LLM: {settings.llm.provider}/{settings.llm.model}", "white")

    # 反思模式从配置文件读取
    enable_reflection = getattr(settings.strategy, "enable_reflection", False)

    agent = None
    try:
        request = create_user_request()

        cprint("正在初始化策略代理...", "white")
        agent = StrategyAgent(
            request,
            enable_reflection=enable_reflection,
        )

        strategy_id = await agent.start()
        cprint(f"策略已启动: {strategy_id}", "green")

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
        cprint(f"系统错误: {e}", "red")
        traceback.print_exc()
    finally:
        cprint("交易系统已停止", "yellow")


if __name__ == "__main__":
    asyncio.run(main())
