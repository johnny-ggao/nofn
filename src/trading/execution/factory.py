"""Factory for creating execution gateways."""

from loguru import logger

from ..models import ExchangeConfig, TradingMode
from .interfaces import BaseExecutionGateway
from .ccxt_trading import CCXTExecutionGateway
from .paper_trading import PaperExecutionGateway


async def create_execution_gateway(config: ExchangeConfig) -> BaseExecutionGateway:
    """Create the appropriate execution gateway based on config.

    Args:
        config: Exchange configuration

    Returns:
        Execution gateway (paper or live)
    """
    if config.trading_mode == TradingMode.VIRTUAL:
        logger.info(f"Creating PaperExecutionGateway for virtual trading (settle_coin={config.settle_coin})")
        return PaperExecutionGateway(
            initial_balance=10000.0,  # Will be overridden by runtime
            fee_bps=config.fee_bps,
            settle_coin=config.settle_coin,
        )

    # Live trading - create CCXT gateway
    logger.info(f"Creating CCXTExecutionGateway for {config.exchange_id}")

    if not config.exchange_id:
        raise ValueError("exchange_id is required for live trading")

    gateway = CCXTExecutionGateway(
        exchange_id=config.exchange_id,
        api_key=config.api_key or "",
        secret_key=config.secret_key or "",
        passphrase=config.passphrase,
        wallet_address=config.wallet_address,
        private_key=config.private_key,
        testnet=config.testnet,
        default_type=config.market_type.value,
        margin_mode=config.margin_mode.value,
        position_mode="oneway",
    )

    # Pre-load markets
    await gateway._get_exchange()

    return gateway
