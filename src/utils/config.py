"""
配置管理模块

提供统一的配置加载和管理功能
"""
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
import yaml


class ExchangeConfig(BaseModel):
    """交易所配置"""
    api_key: str = Field(default="", description="API 密钥")
    api_secret: str = Field(default="", description="API 私钥")
    testnet: bool = Field(default=False, description="是否使用测试网")
    passphrase: Optional[str] = Field(default=None, description="API 密码（某些交易所需要）")


class LLMConfig(BaseModel):
    """LLM 配置"""
    provider: str = Field(default="deepseek", description="LLM 提供商")
    model: str = Field(default="deepseek-chat", description="模型名称")
    api_key: str = Field(default="", description="API 密钥")
    base_url: Optional[str] = Field(default=None, description="API 基础 URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")


class RiskConfig(BaseModel):
    """风控配置"""
    max_position_size: float = Field(default=10000.0, gt=0, description="最大持仓金额（USDT）")
    max_leverage: int = Field(default=10, ge=1, le=125, description="最大杠杆倍数")
    max_daily_trades: int = Field(default=50, gt=0, description="每日最大交易次数")
    enable_risk_check: bool = Field(default=True, description="是否启用风控检查")


class StrategyConfig(BaseModel):
    """策略运行配置"""
    exchange: str = Field(default="hyperliquid", description="交易所")
    symbols: list[str] = Field(default=["BTC/USDC:USDC"], description="交易对列表")
    interval_seconds: int = Field(default=180, gt=0, description="循环间隔（秒）")
    max_iterations: Optional[int] = Field(default=None, description="最大迭代次数，None = 无限")


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = Field(default="INFO", description="日志级别")
    file: str = Field(default="logs/trading.log", description="日志文件路径")
    rotation: str = Field(default="1 day", description="日志轮转周期")
    retention: str = Field(default="30 days", description="日志保留时间")
    compression: str = Field(default="zip", description="日志压缩格式")
    debug: bool = Field(default=False, description="调试模式")


class Config(BaseModel):
    """完整配置"""
    llm: LLMConfig
    exchanges: Dict[str, ExchangeConfig]
    strategy: StrategyConfig
    logging: LoggingConfig
    risk: RiskConfig


class ConfigManager:
    """
    配置管理器

    提供统一的配置加载和访问接口
    """

    _instance: Optional["ConfigManager"] = None
    _config: Optional[Config] = None
    _config_file: Optional[Path] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load_config()

    def load_config(self, config_file: Optional[Path] = None):
        """
        加载配置

        Args:
            config_file: 配置文件路径（默认为 config/config.yaml）
        """
        if config_file is None:
            config_file = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        self._config_file = config_file

        if not config_file.exists():
            raise FileNotFoundError(
                f"配置文件不存在: {config_file}\n"
                "请创建 config/config.yaml 文件并配置相关参数"
            )

        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        self._config = Config(**config_data)

    @property
    def llm(self) -> LLMConfig:
        """获取 LLM 配置"""
        if self._config is None:
            self.load_config()
        return self._config.llm

    @property
    def strategy(self) -> StrategyConfig:
        """获取策略配置"""
        if self._config is None:
            self.load_config()
        return self._config.strategy

    @property
    def logging(self) -> LoggingConfig:
        """获取日志配置"""
        if self._config is None:
            self.load_config()
        return self._config.logging

    @property
    def risk(self) -> RiskConfig:
        """获取风控配置"""
        if self._config is None:
            self.load_config()
        return self._config.risk

    def get_exchange_config(self, exchange_name: str) -> ExchangeConfig:
        """
        获取交易所配置

        Args:
            exchange_name: 交易所名称

        Returns:
            ExchangeConfig: 交易所配置
        """
        if self._config is None:
            self.load_config()

        exchange_name = exchange_name.lower()

        if exchange_name not in self._config.exchanges:
            raise ValueError(
                f"交易所 '{exchange_name}' 未配置\n"
                f"请在 {self._config_file} 中的 exchanges 部分添加配置"
            )

        return self._config.exchanges[exchange_name]

    def get_llm_config(self) -> LLMConfig:
        """获取 LLM 配置（兼容旧接口）"""
        return self.llm


# 全局配置实例
config = ConfigManager()
