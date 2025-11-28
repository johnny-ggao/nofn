"""
交易记忆系统 (基于 Agno 原生能力)

使用 Agno 的原生记忆和会话管理功能：
- SqliteDb: 持久化存储交易案例和会话历史
- MemoryManager: 智能记忆管理和检索
- Session: 会话状态管理
"""
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path

from agno.db.sqlite import SqliteDb
from agno.db.schemas.memory import UserMemory
from agno.memory.manager import MemoryManager
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from termcolor import cprint


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
            for result in self.execution_result[:3]:  # 最多3个结果
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
    交易记忆系统 (基于 Agno)

    使用 Agno 的原生能力：
    1. SqliteDb - 持久化会话和记忆
    2. MemoryManager - 智能记忆管理
    3. session_state - 会话状态管理

    特性：
    - 自动记忆存储和检索
    - 会话历史管理
    - 智能相似案例搜索
    - 记忆摘要生成
    """

    def __init__(
        self,
        db_path: str = "data/agno_trading.db",
        user_id: str = "trading_system",
        model_provider: str = "openai",
        model_id: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_id = user_id

        # 创建 Agno SqliteDb
        self.db = SqliteDb(db_file=str(self.db_path))

        # 创建模型用于记忆管理
        if model_provider == "anthropic":
            self.model = Claude(id=model_id, api_key=api_key)
        else:
            # 对于 DashScope 等不支持 developer 角色的 API，使用 role_map 映射
            # 注意：role_map 需要完整覆盖所有角色，否则会报 KeyError
            role_map = None
            if base_url and ("dashscope" in base_url or "aliyun" in base_url):
                role_map = {
                    "system": "system",      # 保持 system
                    "developer": "system",   # developer -> system
                    "user": "user",
                    "assistant": "assistant",
                    "tool": "tool",
                    "model": "assistant",
                }

            self.model = OpenAIChat(
                id=model_id,
                api_key=api_key,
                base_url=base_url,
                role_map=role_map,
            )

        # 创建 Agno MemoryManager (user_id 在各方法调用时传入)
        self.memory_manager = MemoryManager(
            model=self.model,
            db=self.db,
        )

        # 内部缓存最近案例
        self._recent_cases: List[TradingCase] = []
        self._max_cache_size = 100

        cprint(f"✅ TradingMemory 初始化完成: {db_path}", "green")

    def add_case(self, case: TradingCase) -> str:
        """添加交易案例到记忆"""
        # 转换为记忆内容
        memory_content = case.to_memory_content()

        # 创建 UserMemory 对象 (Agno API 使用 memory_id 而非 id)
        memory = UserMemory(
            memory_id=case.case_id,
            memory=memory_content,
            topics=["trading", "case", case.timestamp.strftime("%Y-%m-%d")],
            input=json.dumps(case.to_dict(), ensure_ascii=False, default=str),
        )

        # 存储到 Agno 记忆系统
        memory_id = self.memory_manager.add_user_memory(
            memory=memory,
            user_id=self.user_id,
        )

        # 添加到本地缓存
        self._recent_cases.append(case)
        if len(self._recent_cases) > self._max_cache_size:
            self._recent_cases = self._recent_cases[-self._max_cache_size:]

        cprint(f"✅ 案例已添加到记忆: {case.case_id}", "green")
        return memory_id or case.case_id

    def search_similar(self, market_conditions: dict, limit: int = 5) -> List[TradingCase]:
        """搜索相似的交易案例"""
        # 构建搜索查询
        query_parts = []
        for symbol, data in market_conditions.get('assets', {}).items():
            trend = data.get('trend', {}).get('direction', '')
            if trend:
                query_parts.append(f"{symbol} {trend}")

        query = " ".join(query_parts) if query_parts else "trading case"

        # 使用 Agno 的记忆搜索
        memories = self.memory_manager.search_user_memories(
            query=query,
            limit=limit,
            user_id=self.user_id,
        )

        # 转换回 TradingCase
        cases = []
        for mem in memories:
            try:
                if mem.input:
                    case_data = json.loads(mem.input)
                    cases.append(TradingCase.from_dict(case_data))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        return cases

    def get_recent_cases(self, days: int = 7, limit: Optional[int] = None) -> List[TradingCase]:
        """获取最近的交易案例"""
        # 从缓存获取
        cutoff = datetime.now() - timedelta(days=days)
        recent = [c for c in self._recent_cases if c.timestamp >= cutoff]

        if limit:
            recent = recent[-limit:]

        return recent

    def get_context(self, market_conditions: Optional[dict] = None, recent_days: int = 7) -> str:
        """
        生成记忆上下文（供 LLM 阅读）

        包括：
        1. 最近表现统计
        2. 相似案例
        3. 核心经验
        """
        lines = ["## 历史记忆", ""]

        # 1. 最近表现
        recent = self.get_recent_cases(recent_days)
        if recent:
            pnls = [c.realized_pnl for c in recent if c.realized_pnl is not None]
            if pnls:
                avg_pnl = sum(pnls) / len(pnls)
                wins = len([p for p in pnls if p > 0])
                win_rate = wins / len(pnls) if pnls else 0

                lines.append(f"### 最近 {recent_days} 天表现")
                lines.append(f"- 交易次数: {len(pnls)}")
                lines.append(f"- 平均盈亏: ${avg_pnl:.2f}")
                lines.append(f"- 胜率: {win_rate * 100:.1f}%")
                lines.append("")

        # 2. 相似案例
        if market_conditions:
            similar = self.search_similar(market_conditions, limit=3)
            if similar:
                lines.append("### 相似历史案例")
                for i, case in enumerate(similar, 1):
                    lines.append(f"#### 案例 {i}")
                    lines.append(f"- 时间: {case.timestamp.strftime('%Y-%m-%d %H:%M')}")
                    if case.realized_pnl is not None:
                        result = "盈利" if case.realized_pnl > 0 else "亏损"
                        lines.append(f"- 结果: {result} ${abs(case.realized_pnl):.2f}")
                    if case.lessons_learned:
                        lines.append(f"- 经验: {', '.join(case.lessons_learned[:2])}")
                    lines.append("")

        # 3. 从 Agno 记忆系统获取用户记忆
        try:
            memories = self.memory_manager.get_user_memories(user_id=self.user_id)
            if memories and len(memories) > 0:
                lines.append("### 核心经验")
                # 提取最近的经验教训
                for mem in memories[-5:]:  # 最近5条记忆
                    if mem.topics and "lesson" in str(mem.topics).lower():
                        lines.append(f"- {mem.memory[:100]}")
                lines.append("")
        except Exception:
            pass

        return "\n".join(lines)

    def get_statistics(self) -> dict:
        """获取统计信息"""
        try:
            stats = self.memory_manager.get_user_memories(user_id=self.user_id)
            memory_count = len(stats) if stats else 0
        except Exception:
            memory_count = 0

        recent = self.get_recent_cases(days=30)
        pnls = [c.realized_pnl for c in recent if c.realized_pnl is not None]

        return {
            'total_memories': memory_count,
            'recent_cases': len(recent),
            'total_pnl': sum(pnls) if pnls else 0,
            'win_rate': len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0,
        }

    async def generate_summary(self, account_info: Optional[dict] = None) -> Optional[str]:
        """使用 Agno Agent 生成记忆摘要"""
        recent = self.get_recent_cases(days=7)
        if len(recent) < 5:
            return None

        # 构建摘要提示
        cases_text = "\n\n".join([c.to_memory_content() for c in recent[-10:]])

        account_text = ""
        if account_info:
            balance = account_info.get('balance', {})
            stats = account_info.get('statistics', {})
            account_text = f"""
## 账户状态
- 余额: ${balance.get('total', 0):.2f}
- 胜率: {stats.get('win_rate', 0) * 100:.1f}%
- 总盈亏: ${stats.get('total_pnl', 0):.2f}
"""

        prompt = f"""
请分析以下交易案例并生成摘要：

{account_text}

{cases_text}

请提供：
1. 关键交易模式（2-3条）
2. 成功策略（2条）
3. 需要避免的错误（2条）
4. 核心经验教训（3条）

保持简洁，每条不超过50字。
"""

        # 使用 Agno Agent 生成摘要
        summary_agent = Agent(
            name="SummaryAgent",
            model=self.model,
            instructions=["你是交易记忆摘要生成器，负责提取交易模式和经验教训。"],
            markdown=False,
        )

        try:
            response = await summary_agent.arun(prompt)
            summary_text = response.content if hasattr(response, 'content') else str(response)

            # 将摘要存储为记忆
            summary_memory = UserMemory(
                memory_id=f"summary_{datetime.now().strftime('%Y%m%d')}",
                memory=summary_text,
                topics=["summary", "weekly", datetime.now().strftime("%Y-%m-%d")],
            )
            self.memory_manager.add_user_memory(
                memory=summary_memory,
                user_id=self.user_id,
            )

            return summary_text
        except Exception as e:
            cprint(f"⚠️  摘要生成失败: {e}", "yellow")
            return None

    def close(self):
        """关闭数据库连接"""
        # Agno SqliteDb 会自动管理连接
        pass
