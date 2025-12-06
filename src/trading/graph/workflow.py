"""LangGraph 交易决策工作流。

定义交易决策图的节点和边，以及 Checkpointer 配置。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from loguru import logger

from .state import (
    TradingState,
    DecisionMemory,
    create_memory_from_result,
    state_to_compose_context,
)


@dataclass
class TradingGraphConfig:
    """交易图配置。"""

    # 数据库路径
    db_path: str = "data/trading_graph.db"

    # 是否启用持久化（测试时可关闭）
    enable_persistence: bool = True


class TradingGraphNodes:
    """交易图节点实现。

    节点是纯函数，接收 State 并返回更新的字段。
    实际的业务逻辑（LLM 调用、交易执行）由外部注入的回调执行。
    """

    def __init__(
        self,
        *,
        decide_callback: Callable[[Dict[str, Any]], Dict[str, Any]],
        execute_callback: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], List[Dict[str, Any]]],
    ):
        """初始化节点。

        Args:
            decide_callback: 决策回调，接收 compose_context，返回 {instructions, rationale}
            execute_callback: 执行回调，接收 (instructions, market_features)，返回 trades
        """
        self._decide_callback = decide_callback
        self._execute_callback = execute_callback

    async def decide(self, state: TradingState) -> Dict[str, Any]:
        """决策节点：调用 LLM 生成交易指令。

        输入：State 中的市场数据和记忆
        输出：instructions, rationale
        """
        # 构建 ComposeContext
        context = state_to_compose_context(state)

        # 调用决策回调
        result = await self._decide_callback(context)

        return {
            "instructions": result.get("instructions", []),
            "rationale": result.get("rationale"),
        }

    async def execute(self, state: TradingState) -> Dict[str, Any]:
        """执行节点：执行交易指令。

        输入：State 中的 instructions
        输出：trades
        """
        instructions = state.get("instructions", [])
        if not instructions:
            return {"trades": []}

        # 提取市场特征用于定价
        market_features = [
            f for f in state.get("features", [])
            if (f.get("meta") or {}).get("source") == "market_snapshot"
        ]

        # 调用执行回调
        trades = await self._execute_callback(instructions, market_features)

        return {"trades": trades}

    def record(self, state: TradingState) -> Dict[str, Any]:
        """记录节点：将决策结果记录到记忆。

        输入：State 中的 instructions, trades, rationale
        输出：memories（增量更新，由 reducer 合并）
        """
        new_memories = create_memory_from_result(
            cycle_index=state["cycle_index"] + 1,
            timestamp_ms=state["timestamp_ms"],
            instructions=state.get("instructions", []),
            trades=state.get("trades", []),
            rationale=state.get("rationale"),
        )

        return {
            "memories": new_memories,  # reducer 会自动合并
            "cycle_index": state["cycle_index"] + 1,
        }


def create_trading_graph(
    *,
    decide_callback: Callable[[Dict[str, Any]], Dict[str, Any]],
    execute_callback: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], List[Dict[str, Any]]],
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> StateGraph:
    """创建交易决策图。

    图结构：
        START -> decide -> execute -> record -> END

    Args:
        decide_callback: 决策回调（async）
        execute_callback: 执行回调（async）
        checkpointer: 状态检查点保存器

    Returns:
        编译后的 StateGraph
    """
    nodes = TradingGraphNodes(
        decide_callback=decide_callback,
        execute_callback=execute_callback,
    )

    # 创建图
    workflow = StateGraph(TradingState)

    # 添加节点
    workflow.add_node("decide", nodes.decide)
    workflow.add_node("execute", nodes.execute)
    workflow.add_node("record", nodes.record)

    # 添加边
    workflow.add_edge(START, "decide")
    workflow.add_edge("decide", "execute")
    workflow.add_edge("execute", "record")
    workflow.add_edge("record", END)

    # 编译（带 checkpointer）
    return workflow.compile(checkpointer=checkpointer)


def get_sqlite_checkpointer_context(
    db_path: str = "data/trading_graph.db",
):
    """获取 SQLite Checkpointer 上下文管理器。

    需要配合 async with 使用：
        async with get_sqlite_checkpointer_context(db_path) as checkpointer:
            graph = create_trading_graph(..., checkpointer=checkpointer)

    Args:
        db_path: 数据库文件路径

    Returns:
        AsyncContextManager[AsyncSqliteSaver]
    """
    # 确保目录存在
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"LangGraph Checkpointer 路径: {db_path}")
    return AsyncSqliteSaver.from_conn_string(db_path)


def get_thread_config(strategy_id: str) -> Dict[str, Any]:
    """获取 LangGraph 线程配置。

    thread_id 使用 strategy_id，这样每个策略有独立的状态。
    """
    return {
        "configurable": {
            "thread_id": strategy_id,
        }
    }
