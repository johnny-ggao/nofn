"""LanceDB 向量存储模块。

提供基于 LanceDB 的语义搜索功能，用于长期记忆的相似性检索。

使用方式：
    # 初始化
    vector_store = LanceDBVectorStore(
        db_path="data/vector_store",
        embedder=OpenAIEmbedder(),
    )

    # 添加记忆
    vector_store.add_memory(strategy_id, memory_id, content, metadata)

    # 语义搜索
    results = vector_store.search(strategy_id, query, limit=5)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING
import time

from loguru import logger

if TYPE_CHECKING:
    import lancedb as lancedb_type

# LanceDB 延迟导入，允许不安装时使用其他功能
try:
    import lancedb
    import pyarrow as pa
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False
    lancedb = None  # type: ignore
    pa = None  # type: ignore


class Embedder(Protocol):
    """嵌入模型协议。"""

    def embed(self, texts: List[str]) -> List[List[float]]:
        """将文本转换为向量。

        Args:
            texts: 文本列表

        Returns:
            向量列表，每个向量是浮点数列表
        """
        ...

    @property
    def dimension(self) -> int:
        """向量维度。"""
        ...


class OpenAIEmbedder:
    """OpenAI 嵌入模型。

    使用 text-embedding-3-small 模型，维度 1536。
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """初始化 OpenAI 嵌入模型。

        Args:
            model: 模型名称
            api_key: API 密钥（可选，默认从环境变量读取）
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("需要安装 openai: pip install openai")

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimension = 1536  # text-embedding-3-small 维度

    def embed(self, texts: List[str]) -> List[List[float]]:
        """将文本转换为向量。"""
        if not texts:
            return []

        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )

        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        """向量维度。"""
        return self._dimension


@dataclass
class VectorSearchResult:
    """向量搜索结果。"""

    memory_id: str
    content: str
    score: float  # 相似度分数（越高越相似）
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """向量存储配置。"""

    # 数据库路径
    db_path: str = "data/vector_store"

    # 表名前缀
    table_prefix: str = "memories"

    # 搜索时返回的最大结果数
    default_limit: int = 10

    # 最小相似度阈值（0-1）
    min_similarity: float = 0.5


class BaseVectorStore(ABC):
    """向量存储基类。"""

    @abstractmethod
    def add_memory(
        self,
        strategy_id: str,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """添加记忆到向量存储。"""
        ...

    @abstractmethod
    def search(
        self,
        strategy_id: str,
        query: str,
        limit: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """语义搜索。"""
        ...

    @abstractmethod
    def delete_memory(self, strategy_id: str, memory_id: str) -> None:
        """删除记忆。"""
        ...

    @abstractmethod
    def clear_strategy(self, strategy_id: str) -> None:
        """清空策略的所有记忆。"""
        ...


class LanceDBVectorStore(BaseVectorStore):
    """LanceDB 向量存储实现。

    特点：
    - 高性能向量搜索
    - 支持混合搜索（向量 + 元数据过滤）
    - 本地存储，无需外部服务
    """

    def __init__(
        self,
        embedder: Embedder,
        config: Optional[VectorStoreConfig] = None,
    ):
        """初始化 LanceDB 向量存储。

        Args:
            embedder: 嵌入模型
            config: 存储配置
        """
        if not LANCEDB_AVAILABLE:
            raise ImportError(
                "需要安装 lancedb: pip install lancedb pyarrow"
            )

        self._embedder = embedder
        self._config = config or VectorStoreConfig()
        self._db: Any = None
        self._tables: Dict[str, Any] = {}

        # 确保目录存在
        Path(self._config.db_path).mkdir(parents=True, exist_ok=True)

    def _get_db(self) -> Any:
        """获取数据库连接。"""
        if self._db is None:
            self._db = lancedb.connect(self._config.db_path)
        return self._db

    def _get_table_name(self, strategy_id: str) -> str:
        """获取策略对应的表名。"""
        # 清理策略 ID，只保留字母数字和下划线
        clean_id = "".join(c if c.isalnum() else "_" for c in strategy_id)
        return f"{self._config.table_prefix}_{clean_id}"

    def _get_or_create_table(self, strategy_id: str) -> Any:
        """获取或创建表。"""
        table_name = self._get_table_name(strategy_id)

        if table_name in self._tables:
            return self._tables[table_name]

        db = self._get_db()

        # 检查表是否存在
        if table_name in db.table_names():
            table = db.open_table(table_name)
        else:
            # 创建新表
            schema = pa.schema([
                pa.field("memory_id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self._embedder.dimension)),
                pa.field("memory_type", pa.string()),
                pa.field("importance", pa.float32()),
                pa.field("total_pnl", pa.float32()),
                pa.field("created_at", pa.float64()),
                pa.field("metadata_json", pa.string()),
            ])
            table = db.create_table(table_name, schema=schema)
            logger.info(f"创建向量表: {table_name}")

        self._tables[table_name] = table
        return table

    def add_memory(
        self,
        strategy_id: str,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """添加记忆到向量存储。

        Args:
            strategy_id: 策略 ID
            memory_id: 记忆 ID
            content: 记忆内容
            metadata: 元数据（type, importance, total_pnl 等）
        """
        import json

        table = self._get_or_create_table(strategy_id)
        metadata = metadata or {}

        # 生成嵌入向量
        vectors = self._embedder.embed([content])
        if not vectors:
            logger.warning(f"无法生成嵌入向量: {memory_id}")
            return

        # 准备数据
        data = [{
            "memory_id": memory_id,
            "content": content,
            "vector": vectors[0],
            "memory_type": metadata.get("type", "unknown"),
            "importance": float(metadata.get("importance", 0.5)),
            "total_pnl": float(metadata.get("total_pnl", 0.0)),
            "created_at": time.time(),
            "metadata_json": json.dumps(metadata),
        }]

        # 检查是否已存在
        try:
            existing = table.search().where(
                f"memory_id = '{memory_id}'", prefilter=True
            ).limit(1).to_list()

            if existing:
                # 删除旧记录
                table.delete(f"memory_id = '{memory_id}'")
        except Exception:
            pass

        # 添加新记录
        table.add(data)
        logger.debug(f"添加向量记忆: {memory_id}, strategy={strategy_id}")

    def search(
        self,
        strategy_id: str,
        query: str,
        limit: int = 10,
        min_similarity: Optional[float] = None,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """语义搜索。

        Args:
            strategy_id: 策略 ID
            query: 查询文本
            limit: 返回数量限制
            min_similarity: 最小相似度（0-1）
            memory_type: 记忆类型过滤
            min_importance: 最小重要性过滤

        Returns:
            搜索结果列表，按相似度降序排列
        """
        import json

        table_name = self._get_table_name(strategy_id)
        db = self._get_db()

        if table_name not in db.table_names():
            return []

        table = self._get_or_create_table(strategy_id)

        # 生成查询向量
        query_vectors = self._embedder.embed([query])
        if not query_vectors:
            return []

        # 构建搜索
        search = table.search(query_vectors[0])

        # 添加过滤条件
        filters = []
        if memory_type:
            filters.append(f"memory_type = '{memory_type}'")
        if min_importance is not None:
            filters.append(f"importance >= {min_importance}")

        if filters:
            search = search.where(" AND ".join(filters), prefilter=True)

        # 执行搜索
        results = search.limit(limit * 2).to_list()  # 多取一些用于过滤

        # 转换结果
        min_sim = min_similarity or self._config.min_similarity
        search_results = []

        for item in results:
            # LanceDB 返回的是 L2 距离，需要转换为相似度
            # 相似度 = 1 / (1 + distance)
            distance = item.get("_distance", 0)
            similarity = 1 / (1 + distance)

            if similarity < min_sim:
                continue

            try:
                metadata = json.loads(item.get("metadata_json", "{}"))
            except Exception:
                metadata = {}

            search_results.append(VectorSearchResult(
                memory_id=item["memory_id"],
                content=item["content"],
                score=similarity,
                metadata=metadata,
            ))

        # 按相似度排序并限制数量
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:limit]

    def delete_memory(self, strategy_id: str, memory_id: str) -> None:
        """删除记忆。"""
        table_name = self._get_table_name(strategy_id)
        db = self._get_db()

        if table_name not in db.table_names():
            return

        table = self._get_or_create_table(strategy_id)
        table.delete(f"memory_id = '{memory_id}'")
        logger.debug(f"删除向量记忆: {memory_id}")

    def clear_strategy(self, strategy_id: str) -> None:
        """清空策略的所有记忆。"""
        table_name = self._get_table_name(strategy_id)
        db = self._get_db()

        if table_name in db.table_names():
            db.drop_table(table_name)
            self._tables.pop(table_name, None)
            logger.info(f"清空向量表: {table_name}")

    def get_stats(self, strategy_id: str) -> Dict[str, Any]:
        """获取策略的向量存储统计。"""
        table_name = self._get_table_name(strategy_id)
        db = self._get_db()

        if table_name not in db.table_names():
            return {"count": 0, "table_exists": False}

        table = self._get_or_create_table(strategy_id)
        count = len(table.to_pandas())

        return {
            "count": count,
            "table_exists": True,
            "table_name": table_name,
        }


class InMemoryVectorStore(BaseVectorStore):
    """内存向量存储（用于测试）。

    使用简单的余弦相似度计算，不依赖 LanceDB。
    """

    def __init__(self, embedder: Embedder):
        """初始化内存向量存储。"""
        self._embedder = embedder
        self._data: Dict[str, Dict[str, Dict[str, Any]]] = {}  # strategy_id -> memory_id -> data

    def add_memory(
        self,
        strategy_id: str,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """添加记忆。"""
        if strategy_id not in self._data:
            self._data[strategy_id] = {}

        vectors = self._embedder.embed([content])
        if not vectors:
            return

        self._data[strategy_id][memory_id] = {
            "content": content,
            "vector": vectors[0],
            "metadata": metadata or {},
        }

    def search(
        self,
        strategy_id: str,
        query: str,
        limit: int = 10,
        min_similarity: Optional[float] = None,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """语义搜索。"""
        if strategy_id not in self._data:
            return []

        query_vectors = self._embedder.embed([query])
        if not query_vectors:
            return []

        query_vec = query_vectors[0]
        results = []

        for memory_id, data in self._data[strategy_id].items():
            # 应用过滤
            metadata = data["metadata"]
            if memory_type and metadata.get("type") != memory_type:
                continue
            if min_importance and metadata.get("importance", 0) < min_importance:
                continue

            # 计算余弦相似度
            similarity = self._cosine_similarity(query_vec, data["vector"])

            if min_similarity and similarity < min_similarity:
                continue

            results.append(VectorSearchResult(
                memory_id=memory_id,
                content=data["content"],
                score=similarity,
                metadata=metadata,
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def delete_memory(self, strategy_id: str, memory_id: str) -> None:
        """删除记忆。"""
        if strategy_id in self._data:
            self._data[strategy_id].pop(memory_id, None)

    def clear_strategy(self, strategy_id: str) -> None:
        """清空策略。"""
        self._data.pop(strategy_id, None)

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """计算余弦相似度。"""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


def create_vector_store(
    embedder: Optional[Embedder] = None,
    config: Optional[VectorStoreConfig] = None,
    use_lancedb: bool = True,
) -> BaseVectorStore:
    """创建向量存储。

    Args:
        embedder: 嵌入模型（默认使用 OpenAI）
        config: 存储配置
        use_lancedb: 是否使用 LanceDB（默认 True）

    Returns:
        向量存储实例
    """
    if embedder is None:
        embedder = OpenAIEmbedder()

    if use_lancedb and LANCEDB_AVAILABLE:
        return LanceDBVectorStore(embedder, config)
    else:
        if use_lancedb and not LANCEDB_AVAILABLE:
            logger.warning("LanceDB 不可用，使用内存向量存储")
        return InMemoryVectorStore(embedder)
