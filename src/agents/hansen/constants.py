from dataclasses import dataclass
from typing import Final

@dataclass
class TradingConstants:
    """交易相关常量"""
    # 仓位管理
    MAX_POSITION_SIZE: Final[float] = 0.2  # 最大单笔仓位
    MIN_POSITION_SIZE: Final[float] = 0.001  # 最小单笔仓位
    DEFAULT_POSITION_SIZE: Final[float] = 0.02  # 默认仓位
    POSITION_REDUCTION_FACTOR: Final[float] = 0.5  # 降级时的仓位缩减系数
    
    # 风险管理
    MAX_DAILY_LOSS: Final[float] = 0.05  # 最大日亏损 5%
    MAX_DRAWDOWN: Final[float] = 0.15  # 最大回撤 15%
    MIN_RISK_REWARD_RATIO: Final[float] = 2.0  # 最小盈亏比 1:2
    MAX_CORRELATION: Final[float] = 0.7  # 最大相关性
    MAX_LEVERAGE: Final[float] = 3.0  # 最大杠杆
    
    # 信号阈值
    MIN_CONFIDENCE: Final[float] = 50.0  # 最小置信度
    HIGH_CONFIDENCE: Final[float] = 80.0  # 高置信度
    SIGNAL_STRENGTH_THRESHOLD: Final[float] = 0.6  # 信号强度阈值
    
    # 技术指标阈值
    RSI_OVERSOLD: Final[float] = 30.0
    RSI_OVERBOUGHT: Final[float] = 70.0
    RSI_NEUTRAL: Final[float] = 50.0
    TREND_STRONG: Final[float] = 0.7
    TREND_WEAK: Final[float] = 0.5
    VOLATILITY_HIGH: Final[float] = 0.5
    VOLATILITY_LOW: Final[float] = 0.3
    VOLUME_SURGE_RATIO: Final[float] = 1.5
    
    # 市场条件
    MIN_MARKET_DEPTH_USDT: Final[float] = 10000.0  # 最小市场深度
    MAX_BID_ASK_SPREAD: Final[float] = 0.002  # 最大买卖价差 0.2%
    MIN_DAILY_VOLUME: Final[float] = 1000.0  # 最小日成交量 USDT
    
    # 重试和超时
    MAX_RETRY_COUNT: Final[int] = 3
    MAX_API_RETRY: Final[int] = 5
    MAX_NETWORK_RETRY: Final[int] = 10
    BACKOFF_FACTOR: Final[float] = 2.0
    MAX_BACKOFF_TIME: Final[int] = 300  # 最大退避时间（秒）
    
    # 时间相关（秒）
    DATA_STALENESS_THRESHOLD: Final[int] = 300  # 数据过期时间 5分钟
    MARKET_INACTIVITY_THRESHOLD: Final[int] = 3600  # 市场不活跃阈值 1小时
    OPTIMIZATION_INTERVAL: Final[int] = 86400  # 优化间隔 24小时
    MONITORING_INTERVAL: Final[int] = 5  # 监控间隔 5秒
    
    # 性能指标阈值
    MIN_WIN_RATE: Final[float] = 0.4  # 最小胜率
    TARGET_WIN_RATE: Final[float] = 0.6  # 目标胜率
    MIN_SHARPE_RATIO: Final[float] = 0.5  # 最小夏普比率
    TARGET_SHARPE_RATIO: Final[float] = 1.5  # 目标夏普比率
    MAX_CONSECUTIVE_LOSSES: Final[int] = 5  # 最大连续亏损次数
    
    # Kelly公式参数
    KELLY_FRACTION: Final[float] = 0.25  # Kelly公式分数（1/4 Kelly）


@dataclass
class SystemConstants:
    """系统相关常量"""
    # API相关
    API_TIMEOUT: Final[int] = 30  # API超时时间（秒）
    WEBSOCKET_PING_INTERVAL: Final[int] = 20  # WebSocket心跳间隔
    WEBSOCKET_PING_TIMEOUT: Final[int] = 10  # WebSocket心跳超时
    
    # 内存和缓存
    MAX_MEMORY_ITEMS: Final[int] = 10000  # 最大记忆条目
    CACHE_TTL: Final[int] = 3600  # 缓存过期时间（秒）
    
    # 日志和监控
    LOG_ROTATION_SIZE: Final[int] = 10485760  # 日志轮转大小 10MB
    MAX_LOG_FILES: Final[int] = 10  # 最大日志文件数
    METRICS_WINDOW_SIZE: Final[int] = 100  # 指标计算窗口大小