"""
日志配置模块

提供统一的日志配置和管理功能
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_config: dict):
    """
    配置日志系统

    Args:
        log_config: 日志配置字典，包含以下字段：
            - level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
            - file: 日志文件路径
            - rotation: 日志轮转周期
            - retention: 日志保留时间
            - compression: 压缩格式
            - debug: 是否启用调试模式
    """
    # 移除默认 handler
    logger.remove()

    # 确定日志级别
    log_level = "DEBUG" if log_config.get("debug", False) else log_config.get("level", "INFO")

    # 添加控制台输出
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # 添加文件输出
    log_file = log_config.get("file", "logs/trading.log")
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="DEBUG",  # 文件始终记录 DEBUG 级别
        rotation=log_config.get("rotation", "1 day"),
        retention=log_config.get("retention", "30 days"),
        compression=log_config.get("compression", "zip"),
    )

    return logger


def get_logger():
    """获取日志实例"""
    return logger
