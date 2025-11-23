"""
ACE - Agentic Context Engineering for Trading

基于 ACE 框架的自进化交易智能体系统
"""

from .agent import ACEAgent
from .models import ContextEntry, ExecutionTrace, Reflection, EntryType
from .storage import ContextStore
from .core import Generator, Reflector, Curator

__version__ = "1.0.0"

__all__ = [
    'ACEAgent',
    'ContextEntry',
    'ExecutionTrace',
    'Reflection',
    'EntryType',
    'ContextStore',
    'Generator',
    'Reflector',
    'Curator',
]
