"""
Trading Knowledge - 交易知识库

基于 Agno Knowledge + LanceDB 的向量知识库，用于存储和检索交易决策历史。

特性：
- 语义搜索：基于相似市场状态检索历史决策
- 持久化存储：本地 LanceDB 存储
- 自动嵌入：使用 OpenAI/兼容 API 生成嵌入向量
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
import json

from termcolor import cprint

from agno.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb
from agno.knowledge.embedder.openai import OpenAIEmbedder


@dataclass
class TradingCase:
    """交易案例数据结构"""

    # 基础信息
    case_id: str
    timestamp: str
    symbol: str

    # 市场状态摘要
    market_summary: str

    # 决策信息
    decision_type: str  # trade, hold, wait
    action: str  # open_long, open_short, close_position, hold, wait
    confidence: int
    reason: str

    # 执行结果
    success: bool
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None

    # 反思
    reflection: Optional[str] = None
    lessons: List[str] = field(default_factory=list)
    quality_score: int = 50

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """转换为可嵌入的文本格式"""
        lines = [
            f"# 交易案例 {self.case_id}",
            f"时间: {self.timestamp}",
            f"标的: {self.symbol}",
            "",
            "## 市场状态",
            self.market_summary,
            "",
            "## 决策",
            f"- 类型: {self.decision_type}",
            f"- 操作: {self.action}",
            f"- 置信度: {self.confidence}%",
            f"- 理由: {self.reason}",
            "",
            "## 结果",
            f"- 成功: {'是' if self.success else '否'}",
        ]

        if self.pnl is not None:
            lines.append(f"- 盈亏: ${self.pnl:.2f} ({self.pnl_percent:.2f}%)" if self.pnl_percent else f"- 盈亏: ${self.pnl:.2f}")

        if self.reflection:
            lines.extend(["", "## 反思", self.reflection])

        if self.lessons:
            lines.extend(["", "## 经验教训"])
            for lesson in self.lessons:
                lines.append(f"- {lesson}")

        lines.append(f"\n质量评分: {self.quality_score}/100")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class TradingKnowledge:
    """
    交易知识库

    使用 Agno Knowledge + LanceDB 实现语义搜索能力。
    """

    def __init__(
        self,
        db_path: str = "data/trading_knowledge",
        table_name: str = "trading_cases",
        embedder_model: str = "text-embedding-3-small",
        embedder_api_key: Optional[str] = None,
        embedder_base_url: Optional[str] = None,
        max_results: int = 5,
    ):
        """
        初始化交易知识库

        Args:
            db_path: LanceDB 存储路径
            table_name: 表名
            embedder_model: 嵌入模型 ID
            embedder_api_key: API Key (可选，默认使用环境变量)
            embedder_base_url: API Base URL (可选，用于兼容 API)
            max_results: 默认检索结果数量
        """
        self.db_path = db_path
        self.table_name = table_name
        self.max_results = max_results

        # 创建 Embedder
        embedder_kwargs = {"id": embedder_model}
        if embedder_api_key:
            embedder_kwargs["api_key"] = embedder_api_key
        if embedder_base_url:
            embedder_kwargs["base_url"] = embedder_base_url

        self.embedder = OpenAIEmbedder(**embedder_kwargs)

        # 创建 LanceDB 向量数据库
        self.vector_db = LanceDb(
            uri=db_path,
            table_name=table_name,
            embedder=self.embedder,
        )

        # 创建 Knowledge 实例
        self.knowledge = Knowledge(
            name="trading_knowledge",
            description="历史交易决策和市场分析记录，用于语义检索相似交易情况",
            vector_db=self.vector_db,
            max_results=max_results,
        )

        cprint(f"✅ TradingKnowledge 初始化完成 (db: {db_path})", "green")

    async def add_case(self, case: TradingCase) -> None:
        """
        添加交易案例到知识库

        Args:
            case: 交易案例
        """
        try:
            # 转换为可嵌入的文本
            text_content = case.to_text()

            # 构建元数据用于过滤
            metadata = {
                "case_id": case.case_id,
                "symbol": case.symbol,
                "action": case.action,
                "success": str(case.success),
                "quality_score": str(case.quality_score),
                "timestamp": case.timestamp,
            }

            # 添加到知识库
            await self.knowledge.add_content_async(
                name=f"case_{case.case_id}",
                text_content=text_content,
                metadata=metadata,
                upsert=True,
            )

            cprint(f"📚 案例已添加到知识库: {case.case_id}", "cyan")

        except Exception as e:
            cprint(f"⚠️  添加案例失败: {e}", "yellow")

    def add_case_sync(self, case: TradingCase) -> None:
        """同步方式添加案例"""
        try:
            text_content = case.to_text()
            metadata = {
                "case_id": case.case_id,
                "symbol": case.symbol,
                "action": case.action,
                "success": str(case.success),
                "quality_score": str(case.quality_score),
                "timestamp": case.timestamp,
            }

            self.knowledge.add_content(
                name=f"case_{case.case_id}",
                text_content=text_content,
                metadata=metadata,
                upsert=True,
            )

            cprint(f"📚 案例已添加到知识库: {case.case_id}", "cyan")

        except Exception as e:
            cprint(f"⚠️  添加案例失败: {e}", "yellow")

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        symbol: Optional[str] = None,
        only_success: bool = False,
        min_quality: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        语义搜索相关交易案例

        Args:
            query: 搜索查询 (市场状态描述、问题等)
            max_results: 最大结果数
            symbol: 限定标的 (可选)
            only_success: 只返回成功案例
            min_quality: 最低质量分数

        Returns:
            相关案例列表
        """
        try:
            # 构建过滤条件
            filters = {}
            if symbol:
                filters["symbol"] = symbol
            if only_success:
                filters["success"] = "True"

            # 执行搜索
            documents = await self.knowledge.async_search(
                query=query,
                max_results=max_results or self.max_results,
                filters=filters if filters else None,
            )

            # 处理结果
            results = []
            for doc in documents:
                # 提取元数据
                meta = doc.meta_data or {}

                # 质量分数过滤
                quality = int(meta.get("quality_score", 0))
                if quality < min_quality:
                    continue

                results.append({
                    "content": doc.content,
                    "name": doc.name,
                    "metadata": meta,
                    "quality_score": quality,
                })

            return results

        except Exception as e:
            cprint(f"⚠️  搜索失败: {e}", "yellow")
            return []

    def search_sync(
        self,
        query: str,
        max_results: Optional[int] = None,
        symbol: Optional[str] = None,
        only_success: bool = False,
        min_quality: int = 0,
    ) -> List[Dict[str, Any]]:
        """同步搜索"""
        try:
            filters = {}
            if symbol:
                filters["symbol"] = symbol
            if only_success:
                filters["success"] = "True"

            documents = self.knowledge.search(
                query=query,
                max_results=max_results or self.max_results,
                filters=filters if filters else None,
            )

            results = []
            for doc in documents:
                meta = doc.meta_data or {}
                quality = int(meta.get("quality_score", 0))
                if quality < min_quality:
                    continue

                results.append({
                    "content": doc.content,
                    "name": doc.name,
                    "metadata": meta,
                    "quality_score": quality,
                })

            return results

        except Exception as e:
            cprint(f"⚠️  搜索失败: {e}", "yellow")
            return []

    async def get_relevant_context(
        self,
        market_summary: str,
        symbol: str,
        max_cases: int = 3,
    ) -> str:
        """
        获取与当前市场状态相关的历史经验上下文

        Args:
            market_summary: 当前市场状态摘要
            symbol: 交易标的
            max_cases: 最大案例数

        Returns:
            格式化的历史经验文本
        """
        # 构建查询
        query = f"{symbol} 市场状态: {market_summary}"

        # 搜索相关案例
        cases = await self.search(
            query=query,
            max_results=max_cases,
            min_quality=40,  # 只返回质量较高的案例
        )

        if not cases:
            return ""

        # 格式化为上下文
        lines = ["## 相关历史经验", ""]
        for i, case in enumerate(cases, 1):
            meta = case.get("metadata", {})
            lines.append(f"### 案例 {i} ({meta.get('timestamp', 'N/A')})")
            lines.append(case.get("content", "")[:500])  # 限制长度
            lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            count = self.vector_db.get_count()
            return {
                "total_cases": count,
                "db_path": self.db_path,
                "table_name": self.table_name,
            }
        except Exception as e:
            return {"error": str(e)}

    def clear(self) -> None:
        """清空知识库"""
        try:
            self.vector_db.drop()
            self.vector_db.create()
            cprint("🗑️  知识库已清空", "yellow")
        except Exception as e:
            cprint(f"⚠️  清空失败: {e}", "red")
