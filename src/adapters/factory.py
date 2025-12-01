"""
交易所适配器工厂

使用工厂模式创建交易所适配器实例
"""
from typing import Dict, Type, Optional, List, Any
from dataclasses import dataclass
from termcolor import cprint

from .base import BaseExchangeAdapter
from .hyperliquid import HyperliquidAdapter
from .binance import BinanceAdapter


@dataclass
class AdapterConfig:
    """适配器配置"""
    name: str
    api_key: str
    api_secret: str
    testnet: bool = False
    margin_type: Optional[str] = None  # 用于 Binance: "USDT" 或 "USDC"


class AdapterFactory:
    """
    交易所适配器工厂

    使用工厂模式创建交易所适配器，简化适配器的创建和管理

    特点：
    - 简单的注册机制
    - 统一的创建接口
    - 自动初始化连接
    - 易于扩展新交易所

    使用示例：
        # 创建 Hyperliquid 适配器
        adapter = await AdapterFactory.create(
            name="hyperliquid",
            api_key=api_key,
            api_secret=api_secret,
        )

        # 创建 Binance USDT 适配器
        adapter = await AdapterFactory.create(
            name="binance",
            api_key=api_key,
            api_secret=api_secret,
            margin_type="USDT",
        )
    """

    # 适配器注册表
    _registry: Dict[str, Type[BaseExchangeAdapter]] = {
        "hyperliquid": HyperliquidAdapter,
        "binance": BinanceAdapter,
        "binance_usdt": BinanceAdapter,
        "binance_usdc": BinanceAdapter,
    }

    @classmethod
    def register(
        cls,
        name: str,
        adapter_class: Type[BaseExchangeAdapter],
    ) -> None:
        """
        注册新的适配器类型

        Args:
            name: 适配器名称（如 "hyperliquid", "binance"）
            adapter_class: 适配器类
        """
        cls._registry[name.lower()] = adapter_class
        cprint(f"✅ 注册交易所适配器: {name}", "green")

    @classmethod
    def list_available(cls) -> List[str]:
        """
        列出所有可用的适配器

        Returns:
            适配器名称列表
        """
        return list(cls._registry.keys())

    @classmethod
    def is_supported(cls, name: str) -> bool:
        """
        检查是否支持指定的交易所

        Args:
            name: 交易所名称

        Returns:
            是否支持
        """
        return name.lower() in cls._registry

    @classmethod
    async def create(
        cls,
        name: str,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        margin_type: Optional[str] = None,
        auto_connect: bool = True,
        **kwargs,
    ) -> BaseExchangeAdapter:
        """
        创建交易所适配器实例

        Args:
            name: 交易所名称 (hyperliquid, binance, binance_usdt, binance_usdc)
            api_key: API 密钥
            api_secret: API 私钥
            testnet: 是否使用测试网
            margin_type: 保证金类型（用于 Binance）
            auto_connect: 是否自动连接
            **kwargs: 其他参数

        Returns:
            初始化完成的适配器实例

        Raises:
            ValueError: 如果交易所不支持
        """
        name_lower = name.lower()

        # 检查是否支持
        if not cls.is_supported(name_lower):
            available = ", ".join(cls.list_available())
            raise ValueError(
                f"不支持的交易所: {name}\n"
                f"可用的交易所: {available}"
            )

        # 获取适配器类
        adapter_class = cls._registry[name_lower]

        # 根据交易所类型创建实例
        adapter = cls._create_adapter_instance(
            name_lower=name_lower,
            adapter_class=adapter_class,
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            margin_type=margin_type,
            **kwargs,
        )

        # 自动初始化连接
        if auto_connect:
            await adapter.initialize()

        return adapter

    @classmethod
    def _create_adapter_instance(
        cls,
        name_lower: str,
        adapter_class: Type[BaseExchangeAdapter],
        api_key: str,
        api_secret: str,
        testnet: bool,
        margin_type: Optional[str],
        **kwargs,
    ) -> BaseExchangeAdapter:
        """
        创建适配器实例（内部方法）

        根据不同的交易所类型，使用不同的参数创建实例
        """
        # Binance 系列
        if name_lower in ["binance", "binance_usdt", "binance_usdc"]:
            # 确定保证金类型
            if name_lower == "binance_usdc":
                effective_margin_type = "USDC"
            elif name_lower == "binance_usdt":
                effective_margin_type = "USDT"
            else:
                effective_margin_type = margin_type or "USDC"

            return BinanceAdapter(
                api_key=api_key,
                api_secret=api_secret,
                margin_type=effective_margin_type,
                testnet=testnet,
                **kwargs,
            )

        # Hyperliquid
        elif name_lower == "hyperliquid":
            return HyperliquidAdapter(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                **kwargs,
            )

        # 其他通用适配器
        else:
            return adapter_class(
                api_key=api_key,
                api_secret=api_secret,
                **kwargs,
            )

    @classmethod
    def get_adapter_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        获取适配器信息

        Args:
            name: 交易所名称

        Returns:
            适配器信息字典
        """
        name_lower = name.lower()

        if not cls.is_supported(name_lower):
            return None

        adapter_class = cls._registry[name_lower]

        # 返回基本信息
        info = {
            "name": name_lower,
            "class": adapter_class.__name__,
            "module": adapter_class.__module__,
        }

        # 添加特定信息
        if name_lower in ["binance", "binance_usdt", "binance_usdc"]:
            info["supported_margin_types"] = ["USDT", "USDC"]
            info["supports_testnet"] = True
        elif name_lower == "hyperliquid":
            info["supported_margin_types"] = ["USDC"]
            info["supports_testnet"] = True

        return info


# 便利函数
async def create_adapter(
    exchange: str,
    api_key: str,
    api_secret: str,
    testnet: bool = False,
    **kwargs,
) -> BaseExchangeAdapter:
    """
    快捷创建适配器的便利函数

    Args:
        exchange: 交易所名称
        api_key: API 密钥
        api_secret: API 私钥
        testnet: 是否使用测试网
        **kwargs: 其他参数

    Returns:
        适配器实例
    """
    return await AdapterFactory.create(
        name=exchange,
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        **kwargs,
    )