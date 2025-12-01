"""
交易记忆系统 (基于 SQLAlchemy + LangChain)

使用标准的 SQL 数据库来存储交易案例，便于学习和自定义
"""
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from termcolor import cprint

Base = declarative_base()


class TradingCaseModel(Base):
    """交易案例表模型"""
    __tablename__ = 'trading_cases'

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(100), unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)

    # 市场条件 (JSON)
    market_conditions = Column(JSON)

    # 决策
    decision = Column(Text)

    # 执行结果 (JSON)
    execution_result = Column(JSON)
    realized_pnl = Column(Float, nullable=True)

    # 反思
    reflection = Column(Text, nullable=True)
    lessons_learned = Column(JSON, nullable=True)  # List[str]

    # 质量评分
    quality_score = Column(Integer, nullable=True)


class TradingSummaryModel(Base):
    """交易摘要表模型"""
    __tablename__ = 'trading_summaries'

    id = Column(Integer, primary_key=True, autoincrement=True)
    summary_id = Column(String(100), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.now)

    # 摘要内容
    summary = Column(Text)

    # 统计信息 (JSON)
    statistics = Column(JSON)

    # 时间范围
    start_date = Column(DateTime)
    end_date = Column(DateTime)


@dataclass
class TradingCase:
    """交易案例（用于记忆存储）"""
    # 市场条件
    market_conditions: dict
    # 决策分析
    decision: str
    # 执行结果
    execution_result: Optional[List[dict]] = None
    realized_pnl: Optional[float] = None
    # 反思
    reflection: Optional[str] = None
    lessons_learned: Optional[List[str]] = None
    quality_score: Optional[int] = None
    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    case_id: Optional[str] = None

    def __post_init__(self):
        if self.case_id is None:
            self.case_id = f"case_{int(self.timestamp.timestamp())}"

    def to_memory_content(self) -> str:
        """转换为记忆内容字符串"""
        lines = [
            f"## 交易案例 {self.case_id}",
            f"时间: {self.timestamp.strftime('%Y-%m-%d %H:%M')}",
            "",
            "### 市场条件",
        ]

        # 简化市场条件
        for symbol, data in self.market_conditions.get('assets', {}).items():
            price = data.get('current_price', 'N/A')
            trend = data.get('trend', {}).get('direction', 'N/A')
            lines.append(f"- {symbol}: ${price}, 趋势: {trend}")

        lines.append("")
        lines.append("### 决策")
        lines.append(self.decision[:500] if self.decision else "N/A")

        if self.execution_result:
            lines.append("")
            lines.append("### 执行结果")
            for result in self.execution_result[:3]:
                signal = result.get('signal', {})
                success = result.get('result', {}).get('success', False)
                lines.append(f"- {signal.get('action', 'N/A')} {signal.get('symbol', '')}: {'成功' if success else '失败'}")

        if self.realized_pnl is not None:
            lines.append(f"- 已实现盈亏: ${self.realized_pnl:.2f}")

        if self.lessons_learned:
            lines.append("")
            lines.append("### 经验教训")
            for lesson in self.lessons_learned[:3]:
                lines.append(f"- {lesson}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'TradingCase':
        """从字典创建"""
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class TradingMemory:
    """
    交易记忆系统 (基于 SQLAlchemy + ChromaDB 向量搜索)

    使用 SQL 数据库存储交易案例，ChromaDB 进行向量相似度搜索
    """

    def __init__(
        self,
        db_path: str = "data/trading_memory.db",
        user_id: str = "default_user",
        vector_store_dir: str = "data/vector_store",
        embedding_provider: str = "dashscope",
        embedding_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        enable_vector_search: bool = True,
    ):
        """
        初始化记忆系统

        Args:
            db_path: 数据库文件路径
            user_id: 用户 ID
            vector_store_dir: 向量存储目录
            embedding_provider: Embedding 提供商 (dashscope, openai, ollama)
            embedding_api_key: Embedding API Key
            embedding_model: Embedding 模型名称 (可选，使用默认值)
            enable_vector_search: 是否启用向量搜索
        """
        self.db_path = db_path
        self.user_id = user_id
        self.enable_vector_search = enable_vector_search

        # 创建数据库目录
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 创建数据库引擎
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)

        # 创建表
        Base.metadata.create_all(self.engine)

        # 创建 Session 工厂
        self.SessionLocal = sessionmaker(bind=self.engine)

        # 初始化向量存储（可选）
        self.vector_store = None
        if enable_vector_search and embedding_api_key:
            try:
                from .vector_store import TradingVectorStore
                self.vector_store = TradingVectorStore(
                    persist_dir=vector_store_dir,
                    collection_name=f"trading_cases_{user_id}",
                    embedding_provider=embedding_provider,
                    embedding_api_key=embedding_api_key,
                    embedding_model=embedding_model,
                )
            except Exception as e:
                cprint(f"⚠️ 向量存储初始化失败，降级为时间排序: {e}", "yellow")
                self.vector_store = None

        cprint(f"✅ TradingMemory 初始化完成 (数据库: {db_path})", "green")
        if self.vector_store:
            cprint(f"   向量搜索: 已启用 (ChromaDB)", "green")
        else:
            cprint(f"   向量搜索: 未启用 (使用时间排序)", "yellow")

    def _get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()

    def add_case(self, case: TradingCase) -> None:
        """添加交易案例（同时保存到 SQL 和向量存储）"""
        session = self._get_session()
        try:
            case_model = TradingCaseModel(
                case_id=case.case_id,
                timestamp=case.timestamp,
                market_conditions=case.market_conditions,
                decision=case.decision,
                execution_result=case.execution_result,
                realized_pnl=case.realized_pnl,
                reflection=case.reflection,
                lessons_learned=case.lessons_learned,
                quality_score=case.quality_score,
            )
            session.add(case_model)
            session.commit()
            cprint(f"✅ 案例已保存到 SQL: {case.case_id}", "green")

            # 同步添加到向量存储
            if self.vector_store:
                try:
                    self.vector_store.add_case(
                        case_id=case.case_id,
                        market_conditions=case.market_conditions,
                        decision=case.decision,
                        lessons_learned=case.lessons_learned,
                        quality_score=case.quality_score,
                        realized_pnl=case.realized_pnl,
                        reflection=case.reflection,
                    )
                except Exception as ve:
                    cprint(f"⚠️ 向量存储保存失败: {ve}", "yellow")

        except Exception as e:
            session.rollback()
            cprint(f"❌ 保存案例失败: {e}", "red")
        finally:
            session.close()

    def get_recent_cases(self, limit: int = 10, days: int = 7) -> List[TradingCase]:
        """获取最近的案例"""
        session = self._get_session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cases = session.query(TradingCaseModel).filter(
                TradingCaseModel.timestamp >= cutoff_date
            ).order_by(TradingCaseModel.timestamp.desc()).limit(limit).all()

            return [self._model_to_case(c) for c in cases]
        finally:
            session.close()

    def search_similar(
        self,
        market_conditions: Dict[str, Any],
        limit: int = 5
    ) -> List[TradingCase]:
        """
        搜索相似案例

        如果启用了向量存储，使用向量相似度搜索
        否则降级为按时间排序的最近案例
        """
        # 优先使用向量搜索
        if self.vector_store:
            try:
                similar_results = self.vector_store.search_similar(
                    market_conditions=market_conditions,
                    limit=limit,
                    min_score=0.3,  # 最小相似度阈值
                )

                if similar_results:
                    # 从 SQL 数据库获取完整案例数据
                    cases = []
                    session = self._get_session()
                    try:
                        for result in similar_results:
                            case_id = result['case_id']
                            case_model = session.query(TradingCaseModel).filter(
                                TradingCaseModel.case_id == case_id
                            ).first()

                            if case_model:
                                case = self._model_to_case(case_model)
                                # 附加相似度信息
                                case.similarity = result.get('similarity', 0)
                                cases.append(case)
                    finally:
                        session.close()

                    if cases:
                        cprint(f"🔍 向量搜索找到 {len(cases)} 个相似案例", "cyan")
                        return cases

            except Exception as e:
                cprint(f"⚠️ 向量搜索失败，降级为时间排序: {e}", "yellow")

        # 降级：返回最近的案例
        return self.get_recent_cases(limit=limit, days=30)

    def get_context(
        self,
        market_conditions: Dict[str, Any],
        recent_days: int = 7,
    ) -> str:
        """
        获取记忆上下文

        返回格式化的字符串，包含最近的交易案例
        """
        cases = self.get_recent_cases(limit=5, days=recent_days)

        if not cases:
            return "暂无历史案例"

        lines = ["# 历史交易案例", ""]

        for i, case in enumerate(cases, 1):
            lines.append(f"## 案例 {i}: {case.case_id}")
            lines.append(f"时间: {case.timestamp.strftime('%Y-%m-%d %H:%M')}")

            if case.realized_pnl is not None:
                result = "盈利" if case.realized_pnl > 0 else "亏损"
                lines.append(f"结果: {result} ${abs(case.realized_pnl):.2f}")

            if case.quality_score:
                lines.append(f"质量评分: {case.quality_score}/100")

            if case.lessons_learned:
                lines.append("经验教训:")
                for lesson in case.lessons_learned[:2]:
                    lines.append(f"  - {lesson}")

            lines.append("")

        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        session = self._get_session()
        try:
            total_cases = session.query(TradingCaseModel).count()

            cutoff_date = datetime.now() - timedelta(days=7)
            recent_cases = session.query(TradingCaseModel).filter(
                TradingCaseModel.timestamp >= cutoff_date
            ).count()

            # 计算平均质量分数
            avg_score_result = session.query(TradingCaseModel.quality_score).filter(
                TradingCaseModel.quality_score.isnot(None)
            ).all()
            avg_score = sum(s[0] for s in avg_score_result) / len(avg_score_result) if avg_score_result else 0

            return {
                'total_cases': total_cases,
                'recent_cases': recent_cases,
                'average_quality_score': avg_score,
            }
        finally:
            session.close()

    def save_summary(self, summary: str, statistics: Dict[str, Any]) -> None:
        """保存摘要"""
        session = self._get_session()
        try:
            summary_model = TradingSummaryModel(
                summary_id=f"summary_{int(datetime.now().timestamp())}",
                created_at=datetime.now(),
                summary=summary,
                statistics=statistics,
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now(),
            )
            session.add(summary_model)
            session.commit()
            cprint("✅ 摘要已保存", "green")
        except Exception as e:
            session.rollback()
            cprint(f"❌ 保存摘要失败: {e}", "red")
        finally:
            session.close()

    def get_latest_summary(self) -> Optional[str]:
        """获取最新摘要"""
        session = self._get_session()
        try:
            summary = session.query(TradingSummaryModel).order_by(
                TradingSummaryModel.created_at.desc()
            ).first()
            return summary.summary if summary else None
        finally:
            session.close()

    def _model_to_case(self, model: TradingCaseModel) -> TradingCase:
        """将数据库模型转换为 TradingCase"""
        return TradingCase(
            case_id=model.case_id,
            timestamp=model.timestamp,
            market_conditions=model.market_conditions or {},
            decision=model.decision or "",
            execution_result=model.execution_result,
            realized_pnl=model.realized_pnl,
            reflection=model.reflection,
            lessons_learned=model.lessons_learned,
            quality_score=model.quality_score,
        )