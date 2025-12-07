"""向量存储测试。

测试 LanceDB 向量搜索功能：
1. 向量存储基本操作（添加、搜索、删除）
2. 混合搜索（向量相似度 + 重要性）
3. 与长期记忆管理器的集成
"""

import math
import pytest
from unittest.mock import MagicMock, patch
from typing import List

from src.trading.graph.vector_store import (
    LANCEDB_AVAILABLE,
    BaseVectorStore,
    InMemoryVectorStore,
    VectorStoreConfig,
    VectorSearchResult,
    create_vector_store,
)
from src.trading.graph.long_term_memory import (
    MemoryType,
    LongTermMemoryConfig,
    LongTermMemoryManager,
    create_in_memory_store,
)


class MockEmbedder:
    """模拟嵌入模型。

    使用简单的词频向量进行测试。
    """

    def __init__(self, dimension: int = 8):
        self._dimension = dimension
        # 预定义一些关键词的向量
        self._word_vectors = {
            "买入": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "卖出": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "持有": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "上涨": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "下跌": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "突破": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "支撑": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "阻力": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """将文本转换为向量。"""
        vectors = []
        for text in texts:
            vec = [0.0] * self._dimension
            for word, word_vec in self._word_vectors.items():
                if word in text:
                    vec = [v1 + v2 for v1, v2 in zip(vec, word_vec)]
            # 归一化
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            else:
                # 默认向量
                vec = [1.0 / math.sqrt(self._dimension)] * self._dimension
            vectors.append(vec)
        return vectors

    @property
    def dimension(self) -> int:
        return self._dimension


class TestInMemoryVectorStore:
    """测试内存向量存储。"""

    def test_add_and_search(self):
        """测试添加和搜索。"""
        embedder = MockEmbedder()
        store = InMemoryVectorStore(embedder)

        # 添加记忆
        store.add_memory(
            strategy_id="test-strategy",
            memory_id="mem-1",
            content="买入时机：价格突破阻力位",
            metadata={"type": "pattern", "importance": 0.8},
        )
        store.add_memory(
            strategy_id="test-strategy",
            memory_id="mem-2",
            content="卖出时机：价格跌破支撑位",
            metadata={"type": "pattern", "importance": 0.7},
        )
        store.add_memory(
            strategy_id="test-strategy",
            memory_id="mem-3",
            content="持有策略：趋势上涨时持有",
            metadata={"type": "lesson", "importance": 0.6},
        )

        # 搜索买入相关
        results = store.search(
            strategy_id="test-strategy",
            query="买入突破",
            limit=2,
        )

        assert len(results) > 0
        assert results[0].memory_id == "mem-1"
        assert "买入" in results[0].content

    def test_search_with_type_filter(self):
        """测试按类型过滤搜索。"""
        embedder = MockEmbedder()
        store = InMemoryVectorStore(embedder)

        store.add_memory(
            strategy_id="test-strategy",
            memory_id="mem-1",
            content="买入模式",
            metadata={"type": "pattern", "importance": 0.8},
        )
        store.add_memory(
            strategy_id="test-strategy",
            memory_id="mem-2",
            content="买入教训",
            metadata={"type": "lesson", "importance": 0.9},
        )

        # 只搜索 pattern 类型
        results = store.search(
            strategy_id="test-strategy",
            query="买入",
            memory_type="pattern",
        )

        assert len(results) == 1
        assert results[0].memory_id == "mem-1"

    def test_search_with_importance_filter(self):
        """测试按重要性过滤搜索。"""
        embedder = MockEmbedder()
        store = InMemoryVectorStore(embedder)

        store.add_memory(
            strategy_id="test-strategy",
            memory_id="mem-1",
            content="低重要性买入",
            metadata={"importance": 0.3},
        )
        store.add_memory(
            strategy_id="test-strategy",
            memory_id="mem-2",
            content="高重要性买入",
            metadata={"importance": 0.9},
        )

        # 只返回高重要性
        results = store.search(
            strategy_id="test-strategy",
            query="买入",
            min_importance=0.5,
        )

        assert len(results) == 1
        assert results[0].memory_id == "mem-2"

    def test_delete_memory(self):
        """测试删除记忆。"""
        embedder = MockEmbedder()
        store = InMemoryVectorStore(embedder)

        store.add_memory(
            strategy_id="test-strategy",
            memory_id="mem-1",
            content="测试内容",
        )

        # 搜索确认存在
        results = store.search("test-strategy", "测试", limit=10)
        assert len(results) == 1

        # 删除
        store.delete_memory("test-strategy", "mem-1")

        # 再次搜索
        results = store.search("test-strategy", "测试", limit=10)
        assert len(results) == 0

    def test_clear_strategy(self):
        """测试清空策略。"""
        embedder = MockEmbedder()
        store = InMemoryVectorStore(embedder)

        store.add_memory("test-strategy", "mem-1", "内容1")
        store.add_memory("test-strategy", "mem-2", "内容2")

        store.clear_strategy("test-strategy")

        results = store.search("test-strategy", "内容", limit=10)
        assert len(results) == 0

    def test_strategy_isolation(self):
        """测试策略隔离。"""
        embedder = MockEmbedder()
        store = InMemoryVectorStore(embedder)

        store.add_memory("strategy-a", "mem-1", "买入策略A")
        store.add_memory("strategy-b", "mem-2", "买入策略B")

        # 只能搜索到对应策略的记忆
        results_a = store.search("strategy-a", "买入", limit=10)
        results_b = store.search("strategy-b", "买入", limit=10)

        assert len(results_a) == 1
        assert results_a[0].memory_id == "mem-1"
        assert len(results_b) == 1
        assert results_b[0].memory_id == "mem-2"


class TestCosineSimiarity:
    """测试余弦相似度计算。"""

    def test_identical_vectors(self):
        """相同向量相似度为 1。"""
        similarity = InMemoryVectorStore._cosine_similarity(
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        )
        assert abs(similarity - 1.0) < 0.0001

    def test_orthogonal_vectors(self):
        """正交向量相似度为 0。"""
        similarity = InMemoryVectorStore._cosine_similarity(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )
        assert abs(similarity) < 0.0001

    def test_opposite_vectors(self):
        """相反向量相似度为 -1。"""
        similarity = InMemoryVectorStore._cosine_similarity(
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        )
        assert abs(similarity - (-1.0)) < 0.0001


class TestLongTermMemoryWithVector:
    """测试长期记忆管理器与向量存储的集成。"""

    def test_manager_with_vector_store(self):
        """测试管理器启用向量搜索。"""
        store = create_in_memory_store()
        vector_store = InMemoryVectorStore(MockEmbedder())

        config = LongTermMemoryConfig(enable_vector_search=True)
        manager = LongTermMemoryManager(store, config, vector_store)

        assert manager.vector_search_enabled

        # 保存记忆
        memory_id = manager.save_memory(
            strategy_id="test-strategy",
            memory_type=MemoryType.PATTERN,
            content="价格突破阻力位时买入",
            importance=0.8,
        )

        assert memory_id is not None

    def test_hybrid_search(self):
        """测试混合搜索。"""
        store = create_in_memory_store()
        vector_store = InMemoryVectorStore(MockEmbedder())

        config = LongTermMemoryConfig(
            enable_vector_search=True,
            hybrid_search_weight=0.5,
        )
        manager = LongTermMemoryManager(store, config, vector_store)

        # 保存多条记忆
        manager.save_memory(
            strategy_id="test-strategy",
            memory_type=MemoryType.PATTERN,
            content="买入时机：突破阻力位",
            importance=0.9,
        )
        manager.save_memory(
            strategy_id="test-strategy",
            memory_type=MemoryType.PATTERN,
            content="买入策略：趋势确认后",
            importance=0.5,
        )
        manager.save_memory(
            strategy_id="test-strategy",
            memory_type=MemoryType.LESSON,
            content="卖出教训：不要恐慌",
            importance=0.7,
        )

        # 搜索买入相关
        results = manager.search_memories(
            strategy_id="test-strategy",
            query="买入突破",
            limit=5,
        )

        assert len(results) >= 1
        # 结果应该包含混合分数
        if results[0].get("_hybrid_score") is not None:
            assert results[0]["_hybrid_score"] > 0

    def test_search_fallback_without_vector(self):
        """测试无向量时回退到重要性排序。"""
        store = create_in_memory_store()

        # 不提供向量存储
        config = LongTermMemoryConfig(enable_vector_search=False)
        manager = LongTermMemoryManager(store, config, None)

        assert not manager.vector_search_enabled

        # 保存记忆
        manager.save_memory(
            strategy_id="test-strategy",
            memory_type=MemoryType.PATTERN,
            content="低重要性",
            importance=0.3,
        )
        manager.save_memory(
            strategy_id="test-strategy",
            memory_type=MemoryType.PATTERN,
            content="高重要性",
            importance=0.9,
        )

        # 搜索应该按重要性排序
        results = manager.search_memories(
            strategy_id="test-strategy",
            query="任意查询",  # 没有向量搜索，query 被忽略
            limit=5,
        )

        assert len(results) == 2
        assert results[0]["importance"] == 0.9
        assert results[1]["importance"] == 0.3


class TestCreateVectorStore:
    """测试向量存储工厂函数。"""

    def test_create_in_memory_store(self):
        """测试创建内存存储。"""
        embedder = MockEmbedder()
        store = create_vector_store(embedder=embedder, use_lancedb=False)

        assert isinstance(store, InMemoryVectorStore)

    @pytest.mark.skipif(not LANCEDB_AVAILABLE, reason="LanceDB not installed")
    def test_create_lancedb_store(self):
        """测试创建 LanceDB 存储（需要安装 lancedb）。"""
        from src.trading.graph.vector_store import LanceDBVectorStore

        embedder = MockEmbedder()
        config = VectorStoreConfig(db_path="/tmp/test_vector_store")
        store = create_vector_store(embedder=embedder, config=config, use_lancedb=True)

        assert isinstance(store, LanceDBVectorStore)


class TestVectorSearchResult:
    """测试向量搜索结果。"""

    def test_result_dataclass(self):
        """测试结果数据类。"""
        result = VectorSearchResult(
            memory_id="mem-123",
            content="测试内容",
            score=0.85,
            metadata={"type": "pattern"},
        )

        assert result.memory_id == "mem-123"
        assert result.content == "测试内容"
        assert result.score == 0.85
        assert result.metadata["type"] == "pattern"

    def test_result_default_metadata(self):
        """测试默认元数据。"""
        result = VectorSearchResult(
            memory_id="mem-123",
            content="测试",
            score=0.5,
        )

        assert result.metadata == {}
