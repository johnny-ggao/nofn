"""
配置文件加载器

从 YAML 配置文件中加载系统配置
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号路径）

        Args:
            key: 配置键，支持点号路径如 "llm.provider"
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    # LLM 配置
    @property
    def llm_provider(self) -> str:
        """LLM 提供商"""
        return self.get('llm.provider', 'openai')

    @property
    def llm_model(self) -> str:
        """LLM 模型"""
        return self.get('llm.model', 'gpt-4o-mini')

    @property
    def llm_api_key(self) -> Optional[str]:
        """LLM API 密钥"""
        return self.get('llm.api_key')

    @property
    def llm_base_url(self) -> Optional[str]:
        """LLM Base URL"""
        return self.get('llm.base_url')

    @property
    def llm_temperature(self) -> float:
        """LLM 温度"""
        return self.get('llm.temperature', 0.7)

    # 交易所配置
    @property
    def exchange(self) -> str:
        """交易所名称"""
        return self.get('strategy.exchange', 'hyperliquid')

    @property
    def symbols(self) -> List[str]:
        """交易对列表"""
        return self.get('strategy.symbols', ['BTC/USDC:USDC'])

    @property
    def interval_seconds(self) -> int:
        """循环间隔（秒）"""
        return self.get('strategy.interval_seconds', 180)

    @property
    def max_iterations(self) -> Optional[int]:
        """最大迭代次数"""
        return self.get('strategy.max_iterations')

    def get_exchange_config(self, exchange_name: str) -> Dict[str, Any]:
        """
        获取指定交易所的配置

        Args:
            exchange_name: 交易所名称

        Returns:
            交易所配置字典
        """
        return self.get(f'exchanges.{exchange_name}', {})

    @property
    def hyperliquid_address(self) -> Optional[str]:
        """Hyperliquid 地址"""
        return self.get('exchanges.hyperliquid.api_key')

    @property
    def hyperliquid_secret(self) -> Optional[str]:
        """Hyperliquid 私钥"""
        return self.get('exchanges.hyperliquid.api_secret')

    # 风险管理配置
    @property
    def max_position_size(self) -> float:
        """最大持仓金额"""
        return self.get('risk.max_position_size', 10000.0)

    @property
    def max_leverage(self) -> int:
        """最大杠杆倍数"""
        return self.get('risk.max_leverage', 10)

    @property
    def max_daily_trades(self) -> int:
        """每日最大交易次数"""
        return self.get('risk.max_daily_trades', 50)

    def __repr__(self) -> str:
        return f"<ConfigLoader(config_path='{self.config_path}')>"