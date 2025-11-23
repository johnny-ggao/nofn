"""
Embedding Service - 文本向量化服务

支持多种 Embedding 提供商：
- OpenAI: text-embedding-3-small (1536维)
- 智谱AI: embedding-2 (1024维)
- Ollama: nomic-embed-text (768维)
- 自定义兼容 OpenAI API 格式的提供商
"""

from typing import List, Optional
from openai import AsyncOpenAI
import asyncio


class EmbeddingService:
    """
    文本向量化服务

    支持多种 Embedding 提供商
    """

    # 预定义提供商配置
    PROVIDERS = {
        "openai": {
            "base_url": None,  # 使用默认
            "model": "text-embedding-3-small",
            "dimensions": 1536
        },
        "zhipu": {  # 智谱AI
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "model": "embedding-2",
            "dimensions": 1024
        },
        "ollama": {  # 本地 Ollama
            "base_url": "http://localhost:11434/v1",
            "model": "nomic-embed-text",
            "dimensions": 768
        },
        "dashscope": {  # 阿里通义
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "text-embedding-v2",
            "dimensions": 1536
        }
    }

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None
    ):
        """
        初始化

        Args:
            api_key: API Key
            base_url: 自定义 API 端点（可选）
            provider: 提供商名称 ("openai", "zhipu", "ollama", "dashscope" 或 "custom")
            model: 模型名称（可选，覆盖默认模型）
        """
        # 获取提供商配置
        if provider in self.PROVIDERS:
            config = self.PROVIDERS[provider]
            actual_base_url = base_url or config["base_url"]
            self.model = model or config["model"]
            self.dimensions = config["dimensions"]
        else:
            # 自定义提供商
            actual_base_url = base_url
            self.model = model or "text-embedding-3-small"
            self.dimensions = 1536

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=actual_base_url
        )
        self.provider = provider

    async def embed(self, text: str) -> List[float]:
        """
        将单个文本转换为向量

        Args:
            text: 输入文本

        Returns:
            向量（维度取决于提供商）
        """
        if not text or not text.strip():
            # 空文本返回零向量
            return [0.0] * self.dimensions

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text[:8000]  # 限制长度
            )
            return response.data[0].embedding

        except Exception as e:
            print(f"⚠️  Embedding 失败 ({self.provider}): {e}")
            # 返回零向量作为降级方案
            return [0.0] * self.dimensions

    async def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        批量向量化

        Args:
            texts: 文本列表
            batch_size: 每批大小

        Returns:
            向量列表
        """
        if not texts:
            return []

        # 分批处理
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=[t[:8000] for t in batch]
                )

                batch_embeddings = [item.embedding for item in response.data]
                results.extend(batch_embeddings)

            except Exception as e:
                print(f"⚠️  Batch embedding 失败: {e}")
                # 降级：逐个处理
                for text in batch:
                    embedding = await self.embed(text)
                    results.append(embedding)

            # 避免速率限制
            if i + batch_size < len(texts):
                await asyncio.sleep(0.5)

        return results

    def embed_sync(self, text: str) -> List[float]:
        """同步版本（用于非异步上下文）"""
        return asyncio.run(self.embed(text))
