"""
适配器管理器

参考 ValueCell 的 AdapterManager 设计，提供：
- 多交易所适配器的统一管理
- 智能故障转移
- 适配器缓存和路由
"""
from typing import Dict, List, Optional, Type
from dataclasses import dataclass
from loguru import logger

from .base import BaseExchangeAdapter
from .hyperliquid import HyperliquidAdapter
from .binance import BinanceAdapter


@dataclass
class AdapterCapability:
    """适配器能力描述"""
    exchange_name: str
    supported_margin_types: List[str]  # ["USDT", "USDC"]
    supports_testnet: bool = True
    supports_hedge_mode: bool = False


@dataclass
class AdapterContext:
    """适配器上下文"""
    name: str
    adapter_class: Type[BaseExchangeAdapter]
    adapter_instance: Optional[BaseExchangeAdapter] = None
    is_connected: bool = False
    capabilities: Optional[AdapterCapability] = None
    error_count: int = 0
    last_error: Optional[str] = None


class AdapterManager:
    """
    适配器管理器

    功能：
    - 统一管理多个交易所适配器
    - 根据配置自动选择适配器
    - 提供故障转移机制
    - 缓存适配器实例

    使用示例:
        manager = AdapterManager()
        manager.register("hyperliquid", HyperliquidAdapter)
        manager.register("binance", BinanceAdapter)

        adapter = await manager.get_adapter("binance", config)
        positions = await adapter.get_positions()
    """

    # 已注册的适配器类
    _registry: Dict[str, Type[BaseExchangeAdapter]] = {
        "hyperliquid": HyperliquidAdapter,
        "binance": BinanceAdapter,
        "binance_usdt": BinanceAdapter,
        "binance_usdc": BinanceAdapter,
    }

    # 适配器能力
    _capabilities: Dict[str, AdapterCapability] = {
        "hyperliquid": AdapterCapability(
            exchange_name="hyperliquid",
            supported_margin_types=["USDC"],
            supports_testnet=True,
            supports_hedge_mode=False,
        ),
        "binance": AdapterCapability(
            exchange_name="binance",
            supported_margin_types=["USDT", "USDC"],
            supports_testnet=True,
            supports_hedge_mode=True,
        ),
    }

    def __init__(self):
        self._contexts: Dict[str, AdapterContext] = {}
        self._active_adapters: Dict[str, BaseExchangeAdapter] = {}

    @classmethod
    def register(
        cls,
        name: str,
        adapter_class: Type[BaseExchangeAdapter],
        capabilities: Optional[AdapterCapability] = None,
    ) -> None:
        """
        注册新的适配器类

        Args:
            name: 适配器名称
            adapter_class: 适配器类
            capabilities: 适配器能力描述
        """
        cls._registry[name.lower()] = adapter_class
        if capabilities:
            cls._capabilities[name.lower()] = capabilities
        logger.info(f"Registered adapter: {name}")

    @classmethod
    def list_adapters(cls) -> List[str]:
        """列出所有已注册的适配器"""
        return list(cls._registry.keys())

    @classmethod
    def get_adapter_class(cls, name: str) -> Optional[Type[BaseExchangeAdapter]]:
        """获取适配器类"""
        return cls._registry.get(name.lower())

    @classmethod
    def get_capabilities(cls, name: str) -> Optional[AdapterCapability]:
        """获取适配器能力"""
        base_name = name.lower().replace("_usdt", "").replace("_usdc", "")
        return cls._capabilities.get(base_name)

    async def create_adapter(
        self,
        name: str,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        margin_type: str = "USDT",
        **kwargs,
    ) -> BaseExchangeAdapter:
        """
        创建并初始化适配器

        Args:
            name: 适配器名称
            api_key: API 密钥
            api_secret: API 私钥
            testnet: 是否使用测试网
            margin_type: 保证金类型
            **kwargs: 其他配置

        Returns:
            初始化后的适配器实例
        """
        name_lower = name.lower()
        adapter_class = self._registry.get(name_lower)

        if not adapter_class:
            raise ValueError(f"Unknown adapter: {name}. Available: {list(self._registry.keys())}")

        # 创建适配器实例
        if name_lower in ["binance", "binance_usdt"]:
            adapter = BinanceAdapter(
                api_key=api_key,
                api_secret=api_secret,
                margin_type="USDT",
                testnet=testnet,
                **kwargs,
            )
        elif name_lower == "binance_usdc":
            adapter = BinanceAdapter(
                api_key=api_key,
                api_secret=api_secret,
                margin_type="USDC",
                testnet=testnet,
                **kwargs,
            )
        elif name_lower == "hyperliquid":
            adapter = HyperliquidAdapter(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                **kwargs,
            )
        else:
            adapter = adapter_class(
                api_key=api_key,
                api_secret=api_secret,
                **kwargs,
            )

        # 初始化连接
        await adapter.initialize()

        # 保存上下文
        context = AdapterContext(
            name=name_lower,
            adapter_class=adapter_class,
            adapter_instance=adapter,
            is_connected=True,
            capabilities=self.get_capabilities(name_lower),
        )
        self._contexts[name_lower] = context
        self._active_adapters[name_lower] = adapter

        logger.info(f"Created and initialized adapter: {name}")
        return adapter

    async def get_adapter(self, name: str) -> Optional[BaseExchangeAdapter]:
        """
        获取已创建的适配器

        Args:
            name: 适配器名称

        Returns:
            适配器实例，或 None
        """
        return self._active_adapters.get(name.lower())

    def get_context(self, name: str) -> Optional[AdapterContext]:
        """获取适配器上下文"""
        return self._contexts.get(name.lower())

    async def close_adapter(self, name: str) -> bool:
        """
        关闭适配器

        Args:
            name: 适配器名称

        Returns:
            是否成功关闭
        """
        name_lower = name.lower()
        adapter = self._active_adapters.get(name_lower)

        if adapter:
            try:
                await adapter.close()
                del self._active_adapters[name_lower]
                if name_lower in self._contexts:
                    self._contexts[name_lower].is_connected = False
                logger.info(f"Closed adapter: {name}")
                return True
            except Exception as e:
                logger.error(f"Error closing adapter {name}: {e}")
                return False

        return False

    async def close_all(self) -> None:
        """关闭所有适配器"""
        for name in list(self._active_adapters.keys()):
            await self.close_adapter(name)

    def record_error(self, name: str, error: str) -> None:
        """记录适配器错误"""
        name_lower = name.lower()
        if name_lower in self._contexts:
            self._contexts[name_lower].error_count += 1
            self._contexts[name_lower].last_error = error
            logger.warning(f"Adapter {name} error #{self._contexts[name_lower].error_count}: {error}")

    def get_fallback_adapter(self, primary: str) -> Optional[str]:
        """
        获取故障转移适配器

        Args:
            primary: 主适配器名称

        Returns:
            备用适配器名称，或 None
        """
        primary_lower = primary.lower()

        # 查找同类型的其他适配器
        primary_capabilities = self.get_capabilities(primary_lower)
        if not primary_capabilities:
            return None

        for name, context in self._contexts.items():
            if name == primary_lower:
                continue
            if context.is_connected and context.capabilities:
                # 检查是否支持相同的保证金类型
                if any(
                    m in context.capabilities.supported_margin_types
                    for m in primary_capabilities.supported_margin_types
                ):
                    return name

        return None


# 全局适配器管理器实例
_global_manager: Optional[AdapterManager] = None


def get_adapter_manager() -> AdapterManager:
    """获取全局适配器管理器"""
    global _global_manager
    if _global_manager is None:
        _global_manager = AdapterManager()
    return _global_manager
