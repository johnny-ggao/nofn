"""
LLM with Prompt Caching Support
支持 Anthropic Claude 的 Prompt Caching 功能
"""
from typing import List, Dict, Optional
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .config import config


def create_llm_with_cache(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    enable_caching: bool = True,
    **kwargs
):
    """
    创建支持 Prompt Caching 的 LLM 实例

    Args:
        provider: LLM 提供商（anthropic 或 openai）
        model: 模型名称
        temperature: 温度参数
        enable_caching: 是否启用 Prompt Caching（仅 Anthropic 支持）
        **kwargs: 其他参数

    Returns:
        LLM 实例
    """
    llm_config = config.llm
    provider = provider or llm_config.provider
    model = model or llm_config.model
    temperature = temperature if temperature is not None else llm_config.temperature
    api_key = llm_config.api_key

    if not api_key:
        raise ValueError(
            f"请在 config/config.yaml 中配置 llm.api_key\n"
            f"当前 provider: {provider}"
        )

    # Anthropic Claude - 支持 Prompt Caching
    if provider.lower() == "anthropic":
        llm_params = {
            "model": model or "claude-3-5-sonnet-20241022",
            "temperature": temperature,
            "anthropic_api_key": api_key,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        # 添加其他参数
        llm_params.update({k: v for k, v in kwargs.items() if k != "max_tokens"})

        return ChatAnthropic(**llm_params)

    # OpenAI / DeepSeek - 不支持 Prompt Caching（但可以用相同的接口）
    else:
        llm_params = {
            "model": model,
            "temperature": temperature,
            "api_key": api_key,
        }

        if llm_config.base_url:
            llm_params["base_url"] = llm_config.base_url

        llm_params.update(kwargs)

        return ChatOpenAI(**llm_params)


def create_cached_messages(
    system_prompt_parts: List[str],
    user_prompt: str,
    enable_caching: bool = True,
    history: Optional[List[Dict]] = None
) -> List:
    """
    创建带有缓存标记的消息列表（Anthropic Claude 专用）

    Args:
        system_prompt_parts: 系统提示词的各个部分（静态部分会被缓存）
            例如: ["核心规则", "交易哲学", "输出格式"]
        user_prompt: 用户提示词（动态内容）
        enable_caching: 是否启用缓存
        history: 对话历史（可选）

    Returns:
        消息列表
    """
    messages = []

    # 系统提示词 - 分段缓存
    if len(system_prompt_parts) == 1:
        # 单段系统提示 - 整体缓存
        messages.append(
            SystemMessage(
                content=system_prompt_parts[0],
                additional_kwargs={
                    "cache_control": {"type": "ephemeral"}
                } if enable_caching else {}
            )
        )
    else:
        # 多段系统提示 - 最后一段缓存（包含前面所有内容）
        for i, part in enumerate(system_prompt_parts):
            is_last = (i == len(system_prompt_parts) - 1)
            messages.append(
                SystemMessage(
                    content=part,
                    additional_kwargs={
                        "cache_control": {"type": "ephemeral"}
                    } if (enable_caching and is_last) else {}
                )
            )

    # 对话历史（如果有）
    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    # 当前用户消息
    messages.append(HumanMessage(content=user_prompt))

    return messages


def create_simple_cached_messages(
    system_prompt: str,
    user_prompt: str,
    enable_caching: bool = True
) -> List:
    """
    创建简单的缓存消息（单个系统提示 + 用户消息）

    Args:
        system_prompt: 完整的系统提示词（会被缓存）
        user_prompt: 用户提示词
        enable_caching: 是否启用缓存

    Returns:
        消息列表
    """
    return [
        SystemMessage(
            content=system_prompt,
            additional_kwargs={
                "cache_control": {"type": "ephemeral"}
            } if enable_caching else {}
        ),
        HumanMessage(content=user_prompt)
    ]


# ==================== Prompt Caching 最佳实践 ====================
"""
Anthropic Prompt Caching 工作原理：

1. 缓存位置：
   - 只能缓存消息列表的"前缀"部分
   - 通常是 SystemMessage 或前几条消息

2. 缓存标记：
   - 使用 additional_kwargs={"cache_control": {"type": "ephemeral"}}
   - 可以标记多个位置，形成"缓存断点"

3. 缓存有效期：
   - 5 分钟内再次使用相同前缀 → 缓存命中
   - 超过 5 分钟 → 缓存失效，重新创建

4. 成本优化：
   - 缓存写入：100% 价格（首次）
   - 缓存读取：10% 价格（5分钟内）
   - 未缓存：100% 价格

5. 最佳实践：
   - 静态内容（规则、格式）→ 缓存
   - 动态内容（市场数据）→ 不缓存
   - 长提示词（>1000 tokens）→ 收益最大
   - 高频调用（<5分钟）→ 收益最大

示例：
```python
messages = [
    SystemMessage(
        content="你是交易智能体...(1800 tokens)",
        additional_kwargs={"cache_control": {"type": "ephemeral"}}  # 缓存
    ),
    SystemMessage(
        content="输出格式：...(780 tokens)",
        additional_kwargs={"cache_control": {"type": "ephemeral"}}  # 缓存
    ),
    HumanMessage(content="当前市场数据...(500 tokens)")  # 不缓存
]

# 首次调用：2580 tokens 写入缓存 + 500 tokens = 3080 tokens
# 后续调用（5分钟内）：2580 tokens × 10% + 500 tokens = 758 tokens
# 节省：75% token 消耗！
```

对于你的项目（每 3 分钟调用一次）：
- System Prompt: 2580 tokens → 始终命中缓存（3分钟 < 5分钟）
- User Prompt: 500-1000 tokens → 每次正常计费
- 理论节省：~75-80% 的 token 成本
"""
