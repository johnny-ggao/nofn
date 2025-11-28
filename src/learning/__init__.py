"""
学习层 (Layer 3) - 完全基于 Agno

混合架构实现：
- Trading Agent (确定性): 快速决策执行
- Analyst Agent (代理式): 深度分析，生成规则

核心组件：
- LearningGraph: 交易学习工作流
- TradingAgents: 决策和即时评估 Agent
- AnalystAgent: 深度分析 Agent（自主检索、多轮反思）
- TradingMemory: 基于 Agno 的记忆系统
- TradingKnowledge: 基于向量数据库的知识库 (语义检索)

这是系统"智能"的核心
"""
from .learning_graph import LearningGraph, TradingState, TradingWorkflow
from .trading_memory import TradingMemory, TradingCase
from .trading_knowledge import TradingKnowledge, TradingCase as KnowledgeCase
from .agents import TradingAgents, DynamicPromptConfig
from .analyst import AnalystAgent, StrategyRule, RuleStore, AnalysisReport

__all__ = [
    'LearningGraph',
    'TradingWorkflow',
    'TradingState',
    'TradingMemory',
    'TradingAgents',
    'TradingCase',
    'TradingKnowledge',
    'KnowledgeCase',
    'DynamicPromptConfig',
    'AnalystAgent',
    'StrategyRule',
    'RuleStore',
    'AnalysisReport',
]
