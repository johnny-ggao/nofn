"""策略模块。

提供策略代理和模板管理。
"""

from .agent import StrategyAgent, run_strategy

# Re-export templates from trading module
from src.trading.templates import get_template_loader, list_templates, TemplateNotFoundError

__all__ = [
    "StrategyAgent",
    "run_strategy",
    "get_template_loader",
    "list_templates",
    "TemplateNotFoundError",
]
