"""
行情数据模型
"""
from typing import Optional, List, Dict, Any
from decimal import Decimal
from pydantic import BaseModel, Field
from datetime import datetime


class Candle(BaseModel):
    """
    K线数据模型
    """
    timestamp: datetime = Field(..., description="K线开始时间")
    open: Decimal = Field(..., description="开盘价")
    high: Decimal = Field(..., description="最高价")
    low: Decimal = Field(..., description="最低价")
    close: Decimal = Field(..., description="收盘价")
    volume: Decimal = Field(..., description="成交量")

    # 可选字段
    quote_volume: Optional[Decimal] = Field(None, description="成交额（计价货币）")
    trades_count: Optional[int] = Field(None, description="成交笔数")
    taker_buy_volume: Optional[Decimal] = Field(None, description="主动买入成交量")
    taker_buy_quote_volume: Optional[Decimal] = Field(None, description="主动买入成交额")

    raw_data: Dict[str, Any] = Field(default_factory=dict, description="原始数据")

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2024-01-01T00:00:00",
                "open": "65000",
                "high": "65500",
                "low": "64800",
                "close": "65200",
                "volume": "100.5"
            }
        }
    }


class Ticker24h(BaseModel):
    """
    24小时行情数据模型
    """
    symbol: str = Field(..., description="交易对")

    # 价格相关
    last_price: Decimal = Field(..., description="最新成交价")
    high_price: Decimal = Field(..., description="24小时最高价")
    low_price: Decimal = Field(..., description="24小时最低价")

    # 成交量相关
    volume: Decimal = Field(..., description="24小时成交量（基础货币）")
    quote_volume: Decimal = Field(..., description="24小时成交额（计价货币）")

    # 价格变化
    price_change: Optional[Decimal] = Field(None, description="24小时价格变化")
    price_change_percent: Optional[Decimal] = Field(None, description="24小时价格变化百分比")

    # 买卖盘
    bid_price: Optional[Decimal] = Field(None, description="最优买一价")
    bid_qty: Optional[Decimal] = Field(None, description="最优买一量")
    ask_price: Optional[Decimal] = Field(None, description="最优卖一价")
    ask_qty: Optional[Decimal] = Field(None, description="最优卖一量")

    # 时间戳
    timestamp: datetime = Field(default_factory=datetime.now, description="数据时间")

    raw_data: Dict[str, Any] = Field(default_factory=dict, description="原始数据")

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "BTC/USDT",
                "last_price": "65000",
                "open_price": "64000",
                "high_price": "66000",
                "low_price": "63500",
                "volume": "1000",
                "quote_volume": "65000000",
                "price_change": "1000",
                "price_change_percent": "1.56"
            }
        }
    }


class FundingRate(BaseModel):
    """
    资金费率模型
    """
    symbol: str = Field(..., description="交易对")

    # 当前资金费率
    funding_rate: Decimal = Field(..., description="当前资金费率")

    # 下次资金费率（如果可用）
    next_funding_rate: Optional[Decimal] = Field(None, description="预测的下次资金费率")

    # 时间相关
    funding_time: datetime = Field(..., description="资金费率结算时间")
    next_funding_time: Optional[datetime] = Field(None, description="下次结算时间")

    # 标记价格和指数价格
    mark_price: Optional[Decimal] = Field(None, description="标记价格")
    index_price: Optional[Decimal] = Field(None, description="指数价格")

    raw_data: Dict[str, Any] = Field(default_factory=dict, description="原始数据")

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "BTC/USDT",
                "funding_rate": "0.0001",
                "funding_time": "2024-01-01T08:00:00",
                "mark_price": "65000",
                "index_price": "64995"
            }
        }
    }


class LatestPrice(BaseModel):
    """
    最新价格信息模型
    """
    symbol: str = Field(..., description="交易对")

    # 最新成交价
    last_price: Decimal = Field(..., description="最新成交价")

    # 标记价格
    mark_price: Optional[Decimal] = Field(None, description="标记价格")

    # 指数价格
    index_price: Optional[Decimal] = Field(None, description="指数价格")

    # 买卖价
    bid_price: Optional[Decimal] = Field(None, description="买一价")
    ask_price: Optional[Decimal] = Field(None, description="卖一价")

    # 24小时涨跌幅（部分交易所支持，如 Binance）
    price_change_percent: Optional[Decimal] = Field(None, description="24小时涨跌幅百分比")

    # 时间戳
    timestamp: datetime = Field(default_factory=datetime.now, description="数据时间")

    raw_data: Dict[str, Any] = Field(default_factory=dict, description="原始数据")

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "BTC/USDT",
                "last_price": "65000",
                "mark_price": "65005",
                "index_price": "64995",
                "bid_price": "64998",
                "ask_price": "65002",
                "price_change_percent": "2.35"
            }
        }
    }


class OrderBook(BaseModel):
    """
    订单簿模型
    """
    symbol: str = Field(..., description="交易对")

    # 买卖盘数据 [[价格, 数量], ...]
    bids: List[List[Decimal]] = Field(default_factory=list, description="买盘")
    asks: List[List[Decimal]] = Field(default_factory=list, description="卖盘")

    # 时间戳
    timestamp: datetime = Field(default_factory=datetime.now, description="数据时间")

    raw_data: Dict[str, Any] = Field(default_factory=dict, description="原始数据")

    @property
    def best_bid(self) -> Optional[Decimal]:
        """最优买一价"""
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        """最优卖一价"""
        return self.asks[0][0] if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        """买卖价差"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """中间价"""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None