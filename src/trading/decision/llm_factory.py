"""LLM factory for creating language model instances.

Provides unified LLM creation for both decision and summarization tasks.
"""

import os
from typing import Any, Callable, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from ..models import LLMModelConfig, SummaryLLMConfig


def create_llm(
    provider: str,
    model_id: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.4,
) -> ChatOpenAI:
    """Create LLM instance based on provider.

    Supports:
    - openai: Direct OpenAI API
    - openrouter: OpenRouter API (OpenAI-compatible)
    - deepseek: DeepSeek API
    - qwen/dashscope: Alibaba Qwen API
    - anthropic: Anthropic Claude API (via OpenRouter or compatible endpoint)

    Args:
        provider: Model provider name
        model_id: Model identifier
        api_key: API key (falls back to environment variables)
        base_url: Custom API base URL (falls back to provider defaults)
        temperature: Model temperature
    """
    # Resolve API key from environment if not provided
    if not api_key:
        env_key_map = {
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_key = env_key_map.get(provider.lower())
        if env_key:
            api_key = os.getenv(env_key)

    if not api_key:
        raise ValueError(
            f"API key not provided for provider '{provider}'. "
            f"Set it in config or via environment variable."
        )

    # Configure base URL - use provided value or fall back to provider defaults
    if not base_url:
        default_urls = {
            "openrouter": "https://openrouter.ai/api/v1",
            "deepseek": "https://api.deepseek.com",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        base_url = default_urls.get(provider.lower())

    # Create LLM
    kwargs: Dict[str, Any] = {
        "model": model_id,
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def create_llm_from_config(config: LLMModelConfig) -> ChatOpenAI:
    """Create LLM instance from LLMModelConfig.

    Args:
        config: LLM model configuration

    Returns:
        ChatOpenAI instance
    """
    return create_llm(
        provider=config.provider,
        model_id=config.model_id,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
    )


def create_summary_llm(config: SummaryLLMConfig) -> Optional[ChatOpenAI]:
    """Create LLM instance for summarization.

    Args:
        config: Summary LLM configuration

    Returns:
        ChatOpenAI instance if enabled, None otherwise
    """
    if not config.enabled:
        return None

    return create_llm(
        provider=config.provider,
        model_id=config.model_id,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
    )


def create_summary_callback(
    llm_config: LLMModelConfig,
) -> Optional[Callable[[str], str]]:
    """Create a summary callback function.

    If summary LLM is enabled, creates a callback that uses the summary LLM.
    Otherwise, returns None (workflow will use rule-based summarization).

    Args:
        llm_config: Main LLM configuration (contains summary_llm config)

    Returns:
        Async callback function or None
    """
    summary_config = llm_config.get_summary_config()

    if not summary_config.enabled:
        logger.debug("Summary LLM not enabled, using rule-based summarization")
        return None

    try:
        summary_llm = create_summary_llm(summary_config)
        if summary_llm is None:
            return None

        logger.info(
            f"Summary LLM initialized: {summary_config.provider}/{summary_config.model_id}"
        )

        async def summarize_callback(memories_text: str) -> str:
            """Summarize memories using LLM."""
            prompt = f"""你是一个交易决策分析助手。请分析以下交易决策历史，生成一个简洁的摘要。

## 决策历史
{memories_text}

## 要求
请用 2-3 句话总结：
1. 主要的交易行为模式（开多/开空/平仓的频率和原因）
2. 执行效果（成功率、盈亏情况）
3. 需要注意的教训或趋势

只输出摘要内容，不要添加其他格式或前缀。"""

            messages = [HumanMessage(content=prompt)]
            response = await summary_llm.ainvoke(messages)
            return response.content.strip()

        return summarize_callback

    except Exception as e:
        logger.warning(f"Failed to create summary LLM, using rule-based: {e}")
        return None
