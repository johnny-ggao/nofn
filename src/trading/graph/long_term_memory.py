"""LangGraph Store 长期记忆管理。

使用 LangGraph 的 Store API 实现跨会话的长期记忆存储和检索。
支持可选的 LanceDB 向量搜索功能。

记忆层次：
    短期记忆 (memories[-10:])     ← LangGraph State + Checkpointer
        ↓
    中期记忆 (summaries[-5:])     ← LLM 压缩的摘要
        ↓ 当 summaries 满时归档
    长期记忆 (Store + Vector)     ← 本模块实现

向量搜索（可选）：
    - 需要安装 lancedb: pip install lancedb pyarrow
    - 需要配置 OpenAI API 用于嵌入向量生成
    - 启用后可进行语义相似性搜索
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import uuid

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import AsyncSqliteStore
from termcolor import cprint

if TYPE_CHECKING:
    from .vector_store import BaseVectorStore


class MemoryType(str, Enum):
    """长期记忆类型。"""

    PATTERN = "pattern"  # 交易模式/规则
    LESSON = "lesson"  # 教训/经验
    CASE = "case"  # 具体案例


@dataclass
class LongTermMemoryConfig:
    """长期记忆配置。"""

    # SQLite 数据库路径（LangGraph Store）
    db_path: str = "data/long_term_memory.db"

    # 向量数据库路径（LanceDB）
    vector_db_path: str = "data/vector_store"

    # 是否启用向量搜索（需要安装 lancedb 和配置 OpenAI API）
    enable_vector_search: bool = False

    # 每个策略最多保留的记忆数量
    max_memories_per_strategy: int = 100

    # 记忆 TTL（秒），None 表示永不过期
    memory_ttl: Optional[float] = None

    # 向量搜索最小相似度阈值（0-1）
    vector_min_similarity: float = 0.5

    # 混合搜索权重（0=纯向量，1=纯重要性）
    hybrid_search_weight: float = 0.5


def get_memory_namespace(
    strategy_id: str,
    memory_type: Optional[MemoryType] = None,
) -> Tuple[str, ...]:
    """获取记忆的命名空间。

    命名空间结构：
        ("strategies", strategy_id, "memories")
        ("strategies", strategy_id, "memories", memory_type)

    Args:
        strategy_id: 策略 ID
        memory_type: 可选的记忆类型

    Returns:
        命名空间元组
    """
    base = ("strategies", strategy_id, "memories")
    if memory_type:
        return base + (memory_type.value,)
    return base


def create_memory_id() -> str:
    """生成唯一的记忆 ID。"""
    return f"mem-{uuid.uuid4().hex[:12]}"


class LongTermMemoryManager:
    """长期记忆管理器。

    封装 LangGraph Store 操作，提供交易场景的记忆管理功能。
    支持可选的向量搜索，实现语义相似性检索。

    使用方式：
        # 基础用法（无向量搜索）
        manager = LongTermMemoryManager(store)

        # 启用向量搜索
        from .vector_store import create_vector_store
        vector_store = create_vector_store()
        config = LongTermMemoryConfig(enable_vector_search=True)
        manager = LongTermMemoryManager(store, config, vector_store)
    """

    def __init__(
        self,
        store: BaseStore,
        config: Optional[LongTermMemoryConfig] = None,
        vector_store: Optional["BaseVectorStore"] = None,
    ):
        """初始化长期记忆管理器。

        Args:
            store: LangGraph Store 实例
            config: 记忆配置
            vector_store: 向量存储（可选，用于语义搜索）
        """
        self._store = store
        self._config = config or LongTermMemoryConfig()
        self._vector_store = vector_store

        # 如果配置启用向量搜索但未提供 vector_store，尝试创建
        if self._config.enable_vector_search and self._vector_store is None:
            self._vector_store = self._try_create_vector_store()

    @property
    def store(self) -> BaseStore:
        """获取底层 Store。"""
        return self._store

    @property
    def vector_store(self) -> Optional["BaseVectorStore"]:
        """获取向量存储（如果启用）。"""
        return self._vector_store

    @property
    def vector_search_enabled(self) -> bool:
        """向量搜索是否已启用。"""
        return self._vector_store is not None

    def _try_create_vector_store(self) -> Optional["BaseVectorStore"]:
        """尝试创建向量存储。"""
        try:
            from .vector_store import create_vector_store, VectorStoreConfig

            config = VectorStoreConfig(
                db_path=self._config.vector_db_path,
                min_similarity=self._config.vector_min_similarity,
            )
            vector_store = create_vector_store(config=config)
            cprint("向量搜索已启用", "white")
            return vector_store
        except Exception as e:
            cprint(f"无法初始化向量存储，向量搜索将被禁用: {e}", "yellow")
            return None

    def save_memory(
        self,
        strategy_id: str,
        memory_type: MemoryType,
        content: str,
        *,
        source_cycles: Optional[List[int]] = None,
        source_summary: Optional[str] = None,
        total_pnl: Optional[float] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """保存一条长期记忆。

        Args:
            strategy_id: 策略 ID
            memory_type: 记忆类型
            content: 记忆内容
            source_cycles: 来源周期范围
            source_summary: 来源摘要
            total_pnl: 相关的总 PnL
            importance: 重要性分数 (0-1)
            metadata: 额外元数据

        Returns:
            记忆 ID
        """
        memory_id = create_memory_id()
        namespace = get_memory_namespace(strategy_id, memory_type)

        value = {
            "type": memory_type.value,
            "content": content,
            "importance": importance,
            "source_cycles": source_cycles or [],
            "source_summary": source_summary,
            "total_pnl": total_pnl,
            "access_count": 0,
            "metadata": metadata or {},
        }

        self._store.put(
            namespace,
            memory_id,
            value,
            ttl=self._config.memory_ttl,
        )

        # 同时写入向量存储（如果启用）
        if self._vector_store is not None:
            try:
                self._vector_store.add_memory(
                    strategy_id=strategy_id,
                    memory_id=memory_id,
                    content=content,
                    metadata={
                        "type": memory_type.value,
                        "importance": importance,
                        "total_pnl": total_pnl,
                    },
                )
            except Exception as e:
                cprint(f"写入向量存储失败: {e}", "yellow")

        cprint(
            f"保存长期记忆: {memory_id}, type={memory_type.value}, "
            f"strategy={strategy_id}, vector={self._vector_store is not None}",
            "magenta"
        )

        return memory_id

    def get_memory(
        self,
        strategy_id: str,
        memory_id: str,
        memory_type: Optional[MemoryType] = None,
    ) -> Optional[Dict[str, Any]]:
        """获取指定的记忆。

        Args:
            strategy_id: 策略 ID
            memory_id: 记忆 ID
            memory_type: 可选的记忆类型（用于优化查找）

        Returns:
            记忆数据，不存在返回 None
        """
        namespace = get_memory_namespace(strategy_id, memory_type)
        item = self._store.get(namespace, memory_id, refresh_ttl=True)

        if item:
            # 更新访问计数
            value = item.value.copy()
            value["access_count"] = value.get("access_count", 0) + 1
            self._store.put(namespace, memory_id, value, ttl=self._config.memory_ttl)
            return value

        return None

    def search_memories(
        self,
        strategy_id: str,
        *,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        min_importance: Optional[float] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """搜索记忆。

        搜索策略：
        1. 如果启用向量搜索且有 query：使用混合搜索（向量相似度 + 重要性）
        2. 否则：从 Store 检索并按重要性排序

        Args:
            strategy_id: 策略 ID
            query: 搜索查询（用于语义搜索，需要启用向量搜索）
            memory_type: 可选的记忆类型过滤
            min_importance: 最小重要性过滤
            limit: 返回数量限制

        Returns:
            匹配的记忆列表
        """
        # 如果启用向量搜索且有查询，使用混合搜索
        if self._vector_store is not None and query:
            return self._hybrid_search(
                strategy_id=strategy_id,
                query=query,
                memory_type=memory_type,
                min_importance=min_importance,
                limit=limit,
            )

        # 否则使用传统的 Store 搜索
        namespace = get_memory_namespace(strategy_id, memory_type)

        # 构建过滤条件
        filter_dict = {}
        if min_importance is not None:
            filter_dict["importance"] = {"$gte": min_importance}

        results = self._store.search(
            namespace,
            filter=filter_dict if filter_dict else None,
            limit=limit,
            refresh_ttl=True,
        )

        memories = []
        for item in results:
            memory = item.value.copy()
            memory["memory_id"] = item.key
            memories.append(memory)

        # 按重要性排序
        memories.sort(key=lambda m: m.get("importance", 0), reverse=True)

        return memories[:limit]

    def _hybrid_search(
        self,
        strategy_id: str,
        query: str,
        memory_type: Optional[MemoryType] = None,
        min_importance: Optional[float] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """混合搜索：结合向量相似度和重要性。

        混合分数 = (1 - weight) * similarity + weight * importance

        Args:
            strategy_id: 策略 ID
            query: 查询文本
            memory_type: 记忆类型过滤
            min_importance: 最小重要性
            limit: 返回数量

        Returns:
            按混合分数排序的记忆列表
        """
        if self._vector_store is None:
            return []

        # 从向量存储搜索
        try:
            vector_results = self._vector_store.search(
                strategy_id=strategy_id,
                query=query,
                limit=limit * 2,  # 多取一些用于混合排序
                min_similarity=self._config.vector_min_similarity,
                memory_type=memory_type.value if memory_type else None,
                min_importance=min_importance,
            )
        except Exception as e:
            cprint(f"向量搜索失败: {e}", "yellow")
            return []

        if not vector_results:
            return []

        # 从 Store 获取完整记忆数据
        memories = []
        weight = self._config.hybrid_search_weight

        for vr in vector_results:
            # 尝试从 Store 获取完整数据
            memory = self.get_memory(strategy_id, vr.memory_id, memory_type)

            if memory is None:
                # 如果 Store 中没有，使用向量结果的元数据
                memory = {
                    "memory_id": vr.memory_id,
                    "content": vr.content,
                    **vr.metadata,
                }

            # 计算混合分数
            importance = memory.get("importance", 0.5)
            hybrid_score = (1 - weight) * vr.score + weight * importance
            memory["_similarity"] = vr.score
            memory["_hybrid_score"] = hybrid_score

            memories.append(memory)

        # 按混合分数排序
        memories.sort(key=lambda m: m.get("_hybrid_score", 0), reverse=True)

        return memories[:limit]

    def delete_memory(
        self,
        strategy_id: str,
        memory_id: str,
        memory_type: Optional[MemoryType] = None,
    ) -> None:
        """删除指定的记忆。

        Args:
            strategy_id: 策略 ID
            memory_id: 记忆 ID
            memory_type: 可选的记忆类型
        """
        namespace = get_memory_namespace(strategy_id, memory_type)
        self._store.delete(namespace, memory_id)

        # 同时从向量存储删除
        if self._vector_store is not None:
            try:
                self._vector_store.delete_memory(strategy_id, memory_id)
            except Exception as e:
                cprint(f"从向量存储删除失败: {e}", "yellow")

        cprint(f"删除长期记忆: {memory_id}, strategy={strategy_id}", "magenta")

    def archive_summary(
        self,
        strategy_id: str,
        summary: Dict[str, Any],
        *,
        extract_patterns: bool = True,
    ) -> List[str]:
        """将摘要归档为长期记忆。

        当中期记忆（summaries）满时，将最旧的摘要转化为长期记忆。

        Args:
            strategy_id: 策略 ID
            summary: DecisionSummary 字典
            extract_patterns: 是否提取模式（需要 LLM）

        Returns:
            创建的记忆 ID 列表
        """
        memory_ids = []

        # 基本信息
        cycle_range = summary.get("cycle_range", (0, 0))
        content = summary.get("content", "")
        total_pnl = summary.get("total_pnl", 0.0)

        # 计算重要性：基于 PnL 和决策数量
        total_decisions = summary.get("total_decisions", 1)
        pnl_factor = min(abs(total_pnl) / 100, 1.0)  # PnL 越大越重要
        importance = 0.5 + 0.5 * pnl_factor

        # 保存为 CASE 类型（具体案例）
        memory_id = self.save_memory(
            strategy_id=strategy_id,
            memory_type=MemoryType.CASE,
            content=content,
            source_cycles=list(range(cycle_range[0], cycle_range[1] + 1)),
            source_summary=content,
            total_pnl=total_pnl,
            importance=importance,
            metadata={
                "total_decisions": total_decisions,
                "executed_count": summary.get("executed_count", 0),
                "time_range": summary.get("time_range"),
            },
        )
        memory_ids.append(memory_id)

        cprint(
            f"归档摘要为长期记忆: {memory_id}, "
            f"cycles={cycle_range}, pnl={total_pnl:.4f}",
            "cyan"
        )

        return memory_ids

    def get_relevant_memories(
        self,
        strategy_id: str,
        *,
        current_context: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """获取与当前决策相关的记忆。

        用于在决策时注入历史经验。

        Args:
            strategy_id: 策略 ID
            current_context: 当前上下文（用于语义搜索）
            limit: 返回数量

        Returns:
            相关记忆列表
        """
        # 搜索所有类型的记忆
        all_memories = []

        for memory_type in MemoryType:
            memories = self.search_memories(
                strategy_id,
                query=current_context,
                memory_type=memory_type,
                min_importance=0.3,
                limit=limit,
            )
            all_memories.extend(memories)

        # 按重要性排序并限制数量
        all_memories.sort(key=lambda m: m.get("importance", 0), reverse=True)
        return all_memories[:limit]

    def format_memories_for_prompt(
        self,
        memories: List[Dict[str, Any]],
    ) -> str:
        """将记忆格式化为 LLM prompt 格式。

        Args:
            memories: 记忆列表

        Returns:
            格式化的文本
        """
        if not memories:
            return ""

        lines = []
        for m in memories:
            memory_type = m.get("type", "unknown")
            content = m.get("content", "")
            importance = m.get("importance", 0)
            pnl = m.get("total_pnl")

            line = f"- [{memory_type}] {content}"
            if pnl:
                line += f" (PnL: {pnl:+.4f})"
            lines.append(line)

        return "\n".join(lines)


def get_sqlite_store_context(
    db_path: str = "data/long_term_memory.db",
):
    """获取 SQLite Store 上下文管理器。

    需要配合 async with 使用：
        async with get_sqlite_store_context(db_path) as store:
            manager = LongTermMemoryManager(store)

    Args:
        db_path: 数据库文件路径

    Returns:
        AsyncContextManager[AsyncSqliteStore]
    """
    # 确保目录存在
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    cprint(f"LangGraph Store 路径: {db_path}", "white")
    return AsyncSqliteStore.from_conn_string(db_path)


def create_in_memory_store() -> InMemoryStore:
    """创建内存 Store（用于测试）。

    Returns:
        InMemoryStore 实例
    """
    return InMemoryStore()
