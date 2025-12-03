"""
策略基类

定义策略的标准接口和配置结构

职责：
- 定义策略配置（时间框架、指标、风控参数）
- 获取和计算市场数据（K线、指标）
- 提供 Prompt 模板
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from pathlib import Path
from enum import Enum

if TYPE_CHECKING:
    from ..adapters.base import BaseExchangeAdapter
    from ..engine.market_snapshot import TimeframeIndicators


class TimeframeType(str, Enum):
    """时间框架类型"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class IndicatorConfig:
    """单个指标配置"""
    name: str                      # 指标名称 (ema, macd, rsi, etc.)
    enabled: bool = True           # 是否启用
    params: Dict[str, Any] = field(default_factory=dict)  # 指标参数


@dataclass
class TimeframeConfig:
    """
    单个时间框架配置

    包含该时间框架使用的指标及其参数
    """
    timeframe: str                 # 时间框架 ("1h", "15m", "5m")
    weight: float                  # 权重 (0.0-1.0)
    purpose: str                   # 用途描述 ("趋势确认", "入场时机", "精确入场")
    candle_limit: int = 100        # K线数量
    indicators: List[IndicatorConfig] = field(default_factory=list)

    def get_indicator(self, name: str) -> Optional[IndicatorConfig]:
        """获取指定指标配置"""
        for ind in self.indicators:
            if ind.name == name:
                return ind
        return None


@dataclass
class StrategyConfig:
    """
    策略配置

    完整定义一个策略的所有参数：
    - 基本信息
    - 时间框架配置
    - 风控参数
    - Prompt 路径
    """
    # 基本信息
    name: str                      # 策略名称
    version: str = "1.0.0"         # 版本号
    description: str = ""          # 策略描述

    # 时间框架配置
    timeframes: List[TimeframeConfig] = field(default_factory=list)

    # 风控参数
    min_confidence: int = 60       # 最小信心度阈值
    min_risk_reward_ratio: float = 2.0  # 最小盈亏比
    max_position_percent: float = 40.0  # 单仓最大资金占比 (%)
    max_leverage: int = 20         # 最大杠杆
    default_risk_percent: float = 2.0   # 默认每笔风险 (%)

    # Prompt 配置
    prompt_path: Optional[str] = None   # Prompt 文件路径

    def get_timeframe(self, tf: str) -> Optional[TimeframeConfig]:
        """获取指定时间框架配置"""
        for tf_config in self.timeframes:
            if tf_config.timeframe == tf:
                return tf_config
        return None

    def get_timeframe_weights(self) -> Dict[str, float]:
        """获取所有时间框架权重"""
        return {tf.timeframe: tf.weight for tf in self.timeframes}

    def validate(self) -> List[str]:
        """
        验证配置有效性

        Returns:
            错误信息列表，空列表表示配置有效
        """
        errors = []

        # 检查时间框架权重之和
        total_weight = sum(tf.weight for tf in self.timeframes)
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"时间框架权重之和应为1.0，当前为 {total_weight:.2f}")

        # 检查风控参数
        if self.min_confidence < 0 or self.min_confidence > 100:
            errors.append(f"min_confidence 应在 0-100 之间，当前为 {self.min_confidence}")

        if self.min_risk_reward_ratio < 1.0:
            errors.append(f"min_risk_reward_ratio 应大于 1.0，当前为 {self.min_risk_reward_ratio}")

        # 检查 Prompt 文件
        if self.prompt_path and not Path(self.prompt_path).exists():
            errors.append(f"Prompt 文件不存在: {self.prompt_path}")

        return errors


class BaseStrategy(ABC):
    """
    策略基类

    定义策略的标准接口，所有具体策略需要继承此类并实现抽象方法
    """

    def __init__(self, config: StrategyConfig):
        """
        初始化策略

        Args:
            config: 策略配置
        """
        self.config = config
        self._prompt_cache: Optional[str] = None

        # 验证配置
        errors = config.validate()
        if errors:
            raise ValueError(f"策略配置无效: {'; '.join(errors)}")

    @property
    def name(self) -> str:
        """策略名称"""
        return self.config.name

    @property
    def version(self) -> str:
        """策略版本"""
        return self.config.version

    def get_prompt(self) -> str:
        """
        获取策略的 Prompt

        从配置的 prompt_path 加载，支持缓存
        """
        if self._prompt_cache is not None:
            return self._prompt_cache

        if self.config.prompt_path:
            prompt_path = Path(self.config.prompt_path)
            if prompt_path.exists():
                self._prompt_cache = prompt_path.read_text(encoding='utf-8')
                return self._prompt_cache

        raise ValueError(f"Prompt 文件不存在: {self.config.prompt_path}")

    @abstractmethod
    def get_indicator_calculator(self):
        """
        获取指标计算器

        返回一个可以根据策略配置计算指标的对象
        """
        pass

    @abstractmethod
    def calculate_indicators(self, ohlcv_data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        根据策略配置计算所有时间框架的指标

        Args:
            ohlcv_data: K线数据，格式 {"1h": OHLCVData, "15m": OHLCVData, "5m": OHLCVData}
            current_price: 当前价格

        Returns:
            各时间框架的指标数据
        """
        pass

    def get_timeframe_list(self) -> List[str]:
        """获取策略使用的时间框架列表"""
        return [tf.timeframe for tf in self.config.timeframes]

    def get_candle_limits(self) -> Dict[str, int]:
        """获取各时间框架需要的K线数量"""
        return {tf.timeframe: tf.candle_limit for tf in self.config.timeframes}

    async def fetch_market_data(
        self,
        adapter: "BaseExchangeAdapter",
        symbol: str,
    ) -> Dict[str, "TimeframeIndicators"]:
        """
        获取并计算市场数据

        策略负责：
        1. 根据配置获取所需的 K 线数据
        2. 计算策略所需的技术指标
        3. 返回各时间框架的指标数据

        Args:
            adapter: 交易所适配器
            symbol: 交易对

        Returns:
            Dict[str, TimeframeIndicators]: 各时间框架的指标数据
        """
        import asyncio
        from ..models import Candle

        # 1. 并发获取所有时间框架的 K 线数据
        candle_tasks = {}
        for tf_config in self.config.timeframes:
            tf = tf_config.timeframe
            limit = tf_config.candle_limit
            candle_tasks[tf] = adapter.get_candles(symbol, tf, limit=limit)

        # 获取当前价格
        price_task = adapter.get_latest_price(symbol)

        # 并发执行
        all_tasks = [price_task] + list(candle_tasks.values())
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # 解析结果
        price_result = results[0]
        if isinstance(price_result, Exception):
            raise ValueError(f"获取价格失败: {price_result}")

        current_price = float(price_result.last_price)

        # 解析 K 线数据
        candles_data = {}
        tf_list = list(candle_tasks.keys())
        for i, tf in enumerate(tf_list):
            candle_result = results[1 + i]
            if isinstance(candle_result, Exception):
                candles_data[tf] = []
            else:
                candles_data[tf] = candle_result

        # 2. 转换为 OHLCV 格式并计算指标
        ohlcv_data = {}
        for tf, candles in candles_data.items():
            if candles and len(candles) >= 20:
                ohlcv_data[tf] = self._candles_to_ohlcv(candles)

        # 3. 调用策略的指标计算方法
        if ohlcv_data:
            return self.calculate_indicators(ohlcv_data, current_price)

        return {}

    @staticmethod
    def _candles_to_ohlcv(candles: List[Any]):
        """将 Candle 列表转换为 OHLCVData"""
        from ..utils.mtf_calculator import OHLCVData
        return OHLCVData(
            open=[float(c.open) for c in candles],
            high=[float(c.high) for c in candles],
            low=[float(c.low) for c in candles],
            close=[float(c.close) for c in candles],
            volume=[float(c.volume) for c in candles],
        )

    def format_indicators(self, asset_data: Any) -> str:
        """
        格式化资产指标数据为文本

        子类可以重写此方法以提供策略特定的指标格式。
        默认使用 AssetData.to_text() 方法。

        Args:
            asset_data: AssetData 实例

        Returns:
            格式化后的指标文本
        """
        return asset_data.to_text()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, version={self.version})>"