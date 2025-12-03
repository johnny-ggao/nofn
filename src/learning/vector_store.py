"""
向量存储模块 - 基于 LangChain + ChromaDB

用于存储和检索交易案例的向量表示，实现相似度搜索
支持多种 Embedding Provider: dashscope (qwen), openai, ollama
"""
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from termcolor import cprint


def create_embeddings(
    provider: str = "dashscope",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Embeddings:
    """
    创建 LangChain Embeddings 实例

    Args:
        provider: embedding 提供商 (dashscope, openai, ollama)
        api_key: API Key
        model: 模型名称 (可选，使用默认值)

    Returns:
        LangChain Embeddings 实例
    """
    if provider == "dashscope":
        from langchain_community.embeddings import DashScopeEmbeddings
        embeddings = DashScopeEmbeddings(
            model=model,
            dashscope_api_key=api_key
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
    return embeddings

class TradingVectorStore:
    """
    交易案例向量存储

    使用 LangChain 的 Chroma 向量存储，
    支持多种 Embedding Provider (DashScope/Qwen, OpenAI, Ollama)
    """

    def __init__(
        self,
        persist_dir: str = "data/vector_store",
        collection_name: str = "trading_cases",
        embedding_provider: str = "dashscope",
        embedding_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        初始化向量存储

        Args:
            persist_dir: 持久化目录
            collection_name: 集合名称
            embedding_provider: Embedding 提供商 (dashscope, openai, ollama)
            embedding_api_key: Embedding API Key
            embedding_base_url: Embedding API URL (可选)
            embedding_model: Embedding 模型名称 (可选，使用默认值)
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider

        # 创建目录
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.embeddings = create_embeddings(
            provider=embedding_provider,
            api_key=embedding_api_key,
            model=embedding_model,
        )

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

        cprint(f"✅ TradingVectorStore 初始化完成 (LangChain + {embedding_provider})", "green")

    @staticmethod
    def _market_conditions_to_text(market_conditions: Dict[str, Any]) -> str:
        """
        将市场条件转换为文本描述

        Args:
            market_conditions: 市场条件字典

        Returns:
            文本描述
        """
        lines = []

        # 提取资产数据
        assets = market_conditions.get('assets', {})
        for symbol, data in assets.items():
            lines.append(f"Symbol: {symbol}")

            # 价格信息
            price_info = data.get('price', {})
            if price_info.get('current'):
                lines.append(f"Price: ${price_info['current']:.2f}")

            # 市场偏向
            market_bias = data.get('market_bias')
            if market_bias:
                lines.append(f"Market Bias: {market_bias}")

            # 多时间框架指标
            timeframes = data.get('timeframes', {})
            for tf_name, tf_data in timeframes.items():
                if tf_data:
                    lines.append(f"Timeframe {tf_name}:")

                    # EMA - 1H uses EMA(7, 21, 55), 15M/5M uses EMA(8, 21, 50)
                    ema = tf_data.get('ema', {})
                    if tf_name == '1h':
                        # 1H级别使用EMA7/21/55
                        if ema.get('ema7') and ema.get('ema21') and ema.get('ema55'):
                            if ema['ema7'] > ema['ema21'] > ema['ema55']:
                                lines.append("  EMA: Bullish alignment (EMA7>21>55)")
                            elif ema['ema7'] < ema['ema21'] < ema['ema55']:
                                lines.append("  EMA: Bearish alignment (EMA7<21<55)")
                            else:
                                lines.append("  EMA: Mixed/Ranging")
                    else:
                        # 15M/5M级别使用EMA8/21/50
                        if ema.get('ema8') and ema.get('ema21') and ema.get('ema50'):
                            if ema['ema8'] > ema['ema21'] > ema['ema50']:
                                lines.append("  EMA: Bullish alignment (EMA8>21>50)")
                            elif ema['ema8'] < ema['ema21'] < ema['ema50']:
                                lines.append("  EMA: Bearish alignment (EMA8<21<50)")
                            else:
                                lines.append("  EMA: Mixed/Ranging")

                    # RSI
                    rsi = tf_data.get('rsi')
                    if rsi:
                        if rsi > 70:
                            lines.append(f"  RSI: {rsi:.1f} (Overbought)")
                        elif rsi < 30:
                            lines.append(f"  RSI: {rsi:.1f} (Oversold)")
                        else:
                            lines.append(f"  RSI: {rsi:.1f} (Neutral)")

                    # MACD
                    macd = tf_data.get('macd', {})
                    if macd.get('histogram'):
                        if macd['histogram'] > 0:
                            lines.append("  MACD: Bullish momentum")
                        else:
                            lines.append("  MACD: Bearish momentum")

                    # ADX
                    adx = tf_data.get('adx', {})
                    if adx.get('value'):
                        if adx['value'] > 25:
                            lines.append(f"  ADX: {adx['value']:.1f} (Strong trend)")
                        else:
                            lines.append(f"  ADX: {adx['value']:.1f} (Weak/Ranging)")

            # 持仓信息
            position = data.get('position', {})
            if position.get('size') and position['size'] > 0:
                lines.append(f"Position: {position.get('side', 'N/A')} {position['size']}")

        # 确保返回非空字符串（embedding API 要求）
        result = "\n".join(lines)
        if not result or not result.strip():
            return "No market data available"
        return result

    def add_case(
        self,
        case_id: str,
        market_conditions: Dict[str, Any],
        decision: str,
        lessons_learned: Optional[List[str]] = None,
        quality_score: Optional[int] = None,
        realized_pnl: Optional[float] = None,
        reflection: Optional[str] = None,
    ) -> None:
        """
        添加交易案例到向量存储

        Args:
            case_id: 案例 ID
            market_conditions: 市场条件
            decision: 决策内容
            lessons_learned: 经验教训
            quality_score: 质量评分
            realized_pnl: 已实现盈亏
            reflection: 反思/评估
        """
        try:
            # 构建用于 embedding 的文本
            market_text = self._market_conditions_to_text(market_conditions)

            # 添加决策和经验
            full_text = f"{market_text}\n\nDecision: {decision}"
            if lessons_learned:
                full_text += f"\n\nLessons: {'; '.join(lessons_learned)}"

            # 构建 LangChain Document
            doc = Document(
                page_content=full_text,
                metadata={
                    "case_id": case_id,
                    "decision": decision[:500] if decision else "",
                    "quality_score": quality_score or 0,
                    "realized_pnl": realized_pnl or 0.0,
                    "reflection": reflection[:500] if reflection else "",
                    "lessons": json.dumps(lessons_learned or []),
                }
            )

            # 使用 LangChain 添加文档
            self.vectorstore.add_documents(
                documents=[doc],
                ids=[case_id],
            )

            cprint(f"✅ 案例已添加到向量存储: {case_id}", "green")

        except Exception as e:
            cprint(f"⚠️ 添加案例到向量存储失败: {e}", "yellow")

    def search_similar(
        self,
        market_conditions: Dict[str, Any],
        limit: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        搜索相似的交易案例

        Args:
            market_conditions: 当前市场条件
            limit: 返回数量限制
            min_score: 最小相似度分数（0-1，越高越严格）

        Returns:
            相似案例列表
        """
        try:
            # 将当前市场条件转换为文本
            query_text = self._market_conditions_to_text(market_conditions)

            # 使用 LangChain 进行相似度搜索（带分数）
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query=query_text,
                k=limit,
            )

            # 解析结果
            similar_cases = []
            for doc, score in results:
                # 跳过相似度低于阈值的案例
                if score < min_score:
                    continue

                metadata = doc.metadata

                # 解析 lessons
                lessons = []
                try:
                    lessons = json.loads(metadata.get('lessons', '[]'))
                except:
                    pass

                similar_cases.append({
                    'case_id': metadata.get('case_id', ''),
                    'similarity': score,
                    'decision': metadata.get('decision', ''),
                    'quality_score': metadata.get('quality_score', 0),
                    'realized_pnl': metadata.get('realized_pnl', 0.0),
                    'reflection': metadata.get('reflection', ''),
                    'lessons_learned': lessons,
                })

            return similar_cases

        except Exception as e:
            cprint(f"⚠️ 向量搜索失败: {e}", "yellow")
            return []

    def delete_case(self, case_id: str) -> bool:
        """
        删除案例

        Args:
            case_id: 案例 ID

        Returns:
            是否删除成功
        """
        try:
            self.vectorstore.delete(ids=[case_id])
            return True
        except Exception as e:
            cprint(f"⚠️ 删除案例失败: {e}", "yellow")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """获取向量存储统计信息"""
        try:
            # 获取底层 Chroma collection
            collection = self.vectorstore._collection
            count = collection.count() if collection else 0
        except:
            count = 0

        return {
            "total_cases": count,
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
        }

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        返回 LangChain Retriever，可用于 LangGraph/LangChain 链

        Args:
            search_kwargs: 搜索参数，如 {"k": 5}

        Returns:
            VectorStoreRetriever
        """
        return self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs or {"k": 5, "score_threshold": 0.3},
        )