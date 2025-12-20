"""K 线特征计算器基类。

定义 K 线特征计算的抽象接口 CandleBasedFeatureComputer。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ...models import Candle, FeatureVector


class CandleBasedFeatureComputer(ABC):
    """K 线特征计算器基类。

    子类从 OHLCV K 线数据计算技术指标。

    使用方法：
    1. 继承此类
    2. 实现 compute_features() 方法
    3. 实现 get_feature_instructions() 方法（可选，提供 LLM 指标说明）
    4. 返回 FeatureVector 列表，包含计算的指标

    示例：
        class MyFeatureComputer(CandleBasedFeatureComputer):
            def compute_features(self, candles, meta=None):
                # 从 K 线数据
                # 计算指标
                # 返回 FeatureVector 列表
                ...

            def get_feature_instructions(self) -> str:
                return "本计算器输出 EMA, RSI 等指标..."
    """

    @abstractmethod
    def compute_features(
        self,
        candles: Optional[List[Candle]] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> List[FeatureVector]:
        """计算 K 线特征。

        Args:
            candles: K 线数据列表
            meta: 可选的元数据，会添加到每个 FeatureVector 中

        Returns:
            FeatureVector 列表，包含计算的指标
        """
        ...

    def get_feature_instructions(self) -> str:
        """获取特征说明，用于 LLM 提示词。

        子类可以覆盖此方法提供指标相关的说明，帮助 LLM 理解数据含义。

        Returns:
            指标说明文本，会被包含在 LLM 提示词中
        """
        return ""
