"""Configuration management for the trading system.

Hybrid configuration system:
- General settings from YAML (config/config.yaml)
- Sensitive credentials from environment variables (.env)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ============================================================================
# Environment Variables (Secrets Only)
# ============================================================================


class SecretsSettings(BaseSettings):
    """Sensitive credentials loaded from environment variables."""

    # LLM API Keys
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    dashscope_api_key: Optional[str] = Field(default=None, alias="DASHSCOPE_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Binance
    binance_api_key: Optional[str] = Field(default=None, alias="BINANCE_API_KEY")
    binance_secret_key: Optional[str] = Field(default=None, alias="BINANCE_SECRET_KEY")

    # OKX
    okx_api_key: Optional[str] = Field(default=None, alias="OKX_API_KEY")
    okx_secret_key: Optional[str] = Field(default=None, alias="OKX_SECRET_KEY")
    okx_passphrase: Optional[str] = Field(default=None, alias="OKX_PASSPHRASE")

    # Bybit
    bybit_api_key: Optional[str] = Field(default=None, alias="BYBIT_API_KEY")
    bybit_secret_key: Optional[str] = Field(default=None, alias="BYBIT_SECRET_KEY")

    # Gate.io
    gate_api_key: Optional[str] = Field(default=None, alias="GATE_API_KEY")
    gate_secret_key: Optional[str] = Field(default=None, alias="GATE_SECRET_KEY")

    # Hyperliquid
    hyperliquid_api_key: Optional[str] = Field(default=None, alias="HYPERLIQUID_API_KEY")
    hyperliquid_secret_key: Optional[str] = Field(default=None, alias="HYPERLIQUID_SECRET_KEY")

    # KuCoin
    kucoin_api_key: Optional[str] = Field(default=None, alias="KUCOIN_API_KEY")
    kucoin_secret_key: Optional[str] = Field(default=None, alias="KUCOIN_SECRET_KEY")
    kucoin_passphrase: Optional[str] = Field(default=None, alias="KUCOIN_PASSPHRASE")

    # MEXC
    mexc_api_key: Optional[str] = Field(default=None, alias="MEXC_API_KEY")
    mexc_secret_key: Optional[str] = Field(default=None, alias="MEXC_SECRET_KEY")

    # Bitget
    bitget_api_key: Optional[str] = Field(default=None, alias="BITGET_API_KEY")
    bitget_secret_key: Optional[str] = Field(default=None, alias="BITGET_SECRET_KEY")
    bitget_passphrase: Optional[str] = Field(default=None, alias="BITGET_PASSPHRASE")

    def get_llm_api_key(self, provider: str) -> Optional[str]:
        """Get API key for the specified LLM provider."""
        provider_lower = provider.lower()
        if provider_lower == "openai":
            return self.openai_api_key
        if provider_lower == "openrouter":
            return self.openrouter_api_key
        if provider_lower == "deepseek":
            return self.deepseek_api_key
        if provider_lower in ("dashscope", "qwen"):
            return self.dashscope_api_key
        if provider_lower in ("anthropic", "claude"):
            return self.anthropic_api_key
        return None

    def get_exchange_credentials(self, exchange_id: str) -> Dict[str, Optional[str]]:
        """Get API credentials for the specified exchange."""
        exchange_lower = exchange_id.lower()

        credentials_map = {
            "binance": {
                "api_key": self.binance_api_key,
                "secret_key": self.binance_secret_key,
                "passphrase": None,
            },
            "okx": {
                "api_key": self.okx_api_key,
                "secret_key": self.okx_secret_key,
                "passphrase": self.okx_passphrase,
            },
            "bybit": {
                "api_key": self.bybit_api_key,
                "secret_key": self.bybit_secret_key,
                "passphrase": None,
            },
            "gate": {
                "api_key": self.gate_api_key,
                "secret_key": self.gate_secret_key,
                "passphrase": None,
            },
            "hyperliquid": {
                "api_key": self.hyperliquid_api_key,
                "secret_key": self.hyperliquid_secret_key,
                "passphrase": None,
            },
            "kucoin": {
                "api_key": self.kucoin_api_key,
                "secret_key": self.kucoin_secret_key,
                "passphrase": self.kucoin_passphrase,
            },
            "mexc": {
                "api_key": self.mexc_api_key,
                "secret_key": self.mexc_secret_key,
                "passphrase": None,
            },
            "bitget": {
                "api_key": self.bitget_api_key,
                "secret_key": self.bitget_secret_key,
                "passphrase": self.bitget_passphrase,
            },
        }

        return credentials_map.get(exchange_lower, {
            "api_key": None,
            "secret_key": None,
            "passphrase": None,
        })

    class Config:
        extra = "ignore"
        env_file = ".env"
        env_file_encoding = "utf-8"


# ============================================================================
# YAML Configuration Models
# ============================================================================


class SummaryLLMConfig(BaseModel):
    """Summary LLM configuration (optional, for memory summarization)."""

    enabled: bool = False  # 是否启用独立的摘要 LLM
    provider: str = "openrouter"
    model: str = "deepseek/deepseek-chat"  # 推荐使用轻量模型
    base_url: Optional[str] = None
    temperature: float = 0.3


class LLMConfig(BaseModel):
    """LLM configuration from YAML."""

    provider: str = "openrouter"
    model: str = "deepseek/deepseek-chat"
    base_url: Optional[str] = None
    temperature: float = 0.4

    # 摘要 LLM 配置（可选）
    summary: Optional[SummaryLLMConfig] = None


class ExchangeConfig(BaseModel):
    """Exchange configuration from YAML."""

    id: str = "binance"
    testnet: bool = False
    market_type: str = "swap"  # spot, future, swap
    margin_mode: str = "cross"  # cross, isolated
    settle_coin: str = "USDT"  # USDT, USDC, USD


class StrategyConfig(BaseModel):
    """Strategy configuration from YAML."""

    name: str = "nofn_strategy"
    template: str = "default"  # Template name: default, aggressive, insane, funding_rate
    symbols: List[str] = Field(default_factory=lambda: ["BTC/USDT:USDT", "ETH/USDT:USDT"])
    initial_capital: float = 10000.0
    max_leverage: float = 10.0
    max_positions: int = 3
    decide_interval: int = 60
    trading_mode: str = "virtual"  # virtual, live


class RiskConfig(BaseModel):
    """Risk management configuration from YAML."""

    max_position_size: float = 10000.0
    max_daily_trades: int = 50
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10


class LoggingConfig(BaseModel):
    """Logging configuration from YAML."""

    level: str = "INFO"
    file: str = "logs/trading.log"
    rotation: str = "1 day"
    retention: str = "30 days"


class YAMLConfig(BaseModel):
    """Complete YAML configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ============================================================================
# Unified Settings
# ============================================================================


class Settings:
    """Combined settings from YAML config and environment variables."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize settings.

        Args:
            config_path: Path to YAML config file. If None, uses default path.
        """
        # Load secrets from environment
        self.secrets = SecretsSettings()

        # Load YAML config
        if config_path is None:
            # Find project root and config file
            project_root = Path(__file__).parent.parent
            yaml_path = project_root / "config" / "config.yaml"
        else:
            yaml_path = Path(config_path)

        self._yaml_config = self._load_yaml(yaml_path)

        # Parse into structured config
        self.config = YAMLConfig(**self._yaml_config)

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ========================================================================
    # Convenience Properties
    # ========================================================================

    @property
    def llm(self) -> LLMConfig:
        """Get LLM configuration."""
        return self.config.llm

    @property
    def exchange(self) -> ExchangeConfig:
        """Get exchange configuration."""
        return self.config.exchange

    @property
    def strategy(self) -> StrategyConfig:
        """Get strategy configuration."""
        return self.config.strategy

    @property
    def risk(self) -> RiskConfig:
        """Get risk configuration."""
        return self.config.risk

    @property
    def logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.config.logging

    # ========================================================================
    # Combined Accessors (YAML + Secrets)
    # ========================================================================

    def get_llm_api_key(self) -> Optional[str]:
        """Get API key for configured LLM provider."""
        return self.secrets.get_llm_api_key(self.llm.provider)

    def get_exchange_credentials(self) -> Dict[str, Optional[str]]:
        """Get API credentials for configured exchange."""
        return self.secrets.get_exchange_credentials(self.exchange.id)

    def get_exchange_api_key(self) -> Optional[str]:
        """Get exchange API key."""
        return self.get_exchange_credentials().get("api_key")

    def get_exchange_secret_key(self) -> Optional[str]:
        """Get exchange secret key."""
        return self.get_exchange_credentials().get("secret_key")

    def get_exchange_passphrase(self) -> Optional[str]:
        """Get exchange passphrase (for OKX, KuCoin, etc.)."""
        return self.get_exchange_credentials().get("passphrase")


# ============================================================================
# Global Instance
# ============================================================================

_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """Get or create settings instance.

    Args:
        config_path: Optional path to YAML config file.

    Returns:
        Settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings(config_path)
    return _settings


def reload_settings(config_path: Optional[str] = None) -> Settings:
    """Force reload settings.

    Args:
        config_path: Optional path to YAML config file.

    Returns:
        New Settings instance.
    """
    global _settings
    _settings = Settings(config_path)
    return _settings


def load_dotenv():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv()
    except ImportError:
        pass
