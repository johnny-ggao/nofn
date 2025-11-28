"""
Agno Agents for Trading System

专用的 Agno Agent 定义：
- DecisionAgent: 交易决策 Agent
- ReflectionAgent: 反思学习 Agent
- SummaryAgent: 记忆摘要 Agent
"""
from typing import Optional, List, Dict, Any
from datetime import datetime

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.tools import tool
from termcolor import cprint

from ..engine.market_snapshot import MarketSnapshot


def create_model(
    provider: str = "openai",
    model_id: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
):
    """创建 Agno 模型"""
    if provider == "anthropic":
        return Claude(id=model_id, api_key=api_key)
    else:
        # 对于 DashScope 等不支持 developer 角色的 API，使用 role_map 映射
        # DashScope 不支持 developer 角色，需要映射为 system
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

        return OpenAIChat(
            id=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            role_map=role_map,
        )


class TradingAgents:
    """
    交易系统 Agents 集合

    包含：
    - decision_agent: 交易决策
    - reflection_agent: 反思学习
    - summary_agent: 记忆摘要
    """

    def __init__(
        self,
        db: Optional[SqliteDb] = None,
        model_provider: str = "openai",
        model_id: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        system_prompt_path: Optional[str] = None,
    ):
        self.db = db
        self.model = create_model(
            provider=model_provider,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

        # 加载系统提示词
        if system_prompt_path:
            from pathlib import Path
            self.system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")
        else:
            self.system_prompt = self._default_decision_prompt()

        # 创建各个 Agent
        self._create_agents()

    def _default_decision_prompt(self) -> str:
        """默认决策提示词"""
        return """你是一个专业的加密货币交易分析师和决策者。

你的职责是：
1. 分析市场数据和技术指标
2. 结合历史经验做出交易决策
3. 提供清晰的买入/卖出/持有建议

决策原则：
- 风险控制优先，每笔交易必须设置止损
- 趋势跟踪，顺势而为
- 仓位管理，不过度交易
- 保持客观，避免情绪化决策

输出格式要求：
- 先进行市场分析
- 然后给出具体的交易建议（JSON格式）
"""

    def _create_agents(self):
        """创建所有 Agents"""
        # 决策 Agent - 使用会话历史和记忆
        self.decision_agent = Agent(
            name="TradingDecisionAgent",
            model=self.model,
            db=self.db,
            session_id="decision_session",
            instructions=[self.system_prompt],
            add_history_to_context=True,
            num_history_runs=3,
            enable_session_summaries=True,
            markdown=False,
        )

        # 反思 Agent
        self.reflection_agent = Agent(
            name="ReflectionAgent",
            model=self.model,
            db=self.db,
            session_id="reflection_session",
            instructions=[
                "你是一个交易系统的反思模块，负责分析交易决策和结果。",
                "你的任务是：",
                "1. 评估决策质量",
                "2. 分析执行结果",
                "3. 提取经验教训",
                "4. 提供改进建议",
                "",
                "请保持客观、简洁，专注于可操作的洞察。",
            ],
            add_history_to_context=True,
            num_history_runs=5,
            markdown=False,
        )

        # 摘要 Agent
        self.summary_agent = Agent(
            name="SummaryAgent",
            model=self.model,
            instructions=[
                "你是交易记忆摘要生成器。",
                "你的任务是分析交易案例，提取：",
                "1. 关键交易模式",
                "2. 成功策略",
                "3. 失败教训",
                "4. 核心经验",
                "",
                "保持简洁，每条不超过50字。",
            ],
            markdown=False,
        )

        cprint("✅ TradingAgents 创建完成", "green")

    async def make_decision(
        self,
        market_snapshot: MarketSnapshot,
        memory_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        使用 DecisionAgent 做出交易决策

        Returns:
            {
                'decision_type': 'trade|hold|wait',
                'signals': [...],
                'analysis': '...',
            }
        """
        # 构建上下文
        context = self._build_decision_context(market_snapshot, memory_context)

        # 调用 Agent
        response = await self.decision_agent.arun(context)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # 解析响应
        return self._parse_decision_response(response_text)

    async def reflect(
        self,
        decision: Dict[str, Any],
        execution_results: List[Dict],
        account_info: Optional[Dict] = None,
        market_snapshot: Optional[MarketSnapshot] = None,
    ) -> Dict[str, Any]:
        """
        使用 ReflectionAgent 进行反思

        Returns:
            {
                'reflection': '...',
                'lessons': [...],
                'quality_score': 0-100,
            }
        """
        # 构建反思上下文
        context = self._build_reflection_context(
            decision, execution_results, account_info, market_snapshot
        )

        # 调用 Agent
        response = await self.reflection_agent.arun(context)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # 解析响应
        return self._parse_reflection_response(response_text)

    async def generate_summary(
        self,
        cases_text: str,
        account_info: Optional[Dict] = None,
    ) -> str:
        """使用 SummaryAgent 生成摘要"""
        account_text = ""
        if account_info:
            balance = account_info.get('balance', {})
            stats = account_info.get('statistics', {})
            account_text = f"""
## 账户状态
- 余额: ${balance.get('total', 0):.2f}
- 可用: ${balance.get('available', 0):.2f}
- 胜率: {stats.get('win_rate', 0) * 100:.1f}%
"""

        prompt = f"""
请分析以下交易记录并生成摘要：

{account_text}

{cases_text}

请输出：
1. **关键模式** (2-3条)
2. **成功策略** (2条)
3. **避免错误** (2条)
4. **核心经验** (3条)
"""

        response = await self.summary_agent.arun(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def _build_decision_context(
        self,
        snapshot: MarketSnapshot,
        memory: Optional[str],
    ) -> str:
        """构建决策上下文"""
        lines = [
            "请分析以下市场情况并做出交易决策：",
            "",
            "=" * 60,
            snapshot.to_text(),
            "=" * 60,
        ]

        if memory:
            lines.extend([
                "",
                "## 历史记忆",
                memory,
                "=" * 60,
            ])

        lines.extend([
            "",
            "## 决策要求",
            "",
            "请以以下 JSON 格式输出决策：",
            "```json",
            "{",
            '  "decision_type": "trade | hold | wait",',
            '  "signals": [',
            '    {',
            '      "action": "open_long | open_short | close_position | hold | wait",',
            '      "symbol": "BTC/USDC:USDC",',
            '      "amount": 0.001,',
            '      "leverage": 3,',
            '      "stop_loss": 88000.0,',
            '      "take_profit": 96000.0,',
            '      "confidence": 85,',
            '      "reason": "原因说明"',
            '    }',
            '  ]',
            "}",
            "```",
        ])

        return "\n".join(lines)

    def _build_reflection_context(
        self,
        decision: Dict[str, Any],
        execution_results: List[Dict],
        account_info: Optional[Dict],
        market_snapshot: Optional[MarketSnapshot],
    ) -> str:
        """构建反思上下文"""
        lines = ["请反思以下交易过程：", "", "## 账户状态"]

        if account_info:
            balance = account_info.get('balance', {})
            stats = account_info.get('statistics', {})
            lines.extend([
                f"- 账户余额: ${balance.get('total', 0):.2f}",
                f"- 可用资金: ${balance.get('available', 0):.2f}",
                f"- 胜率: {stats.get('win_rate', 0) * 100:.1f}%",
                f"- 总盈亏: ${stats.get('total_pnl', 0):.2f}",
            ])
        else:
            lines.append("（账户信息不可用）")

        lines.extend([
            "",
            "## 市场条件",
            market_snapshot.to_text() if market_snapshot else "N/A",
            "",
            "## 决策",
            decision.get('analysis', 'N/A'),
            "",
            "## 执行结果",
        ])

        for result in execution_results:
            signal = result.get('signal', {})
            success = result.get('result', {}).get('success', False)
            action = signal.get('action', 'N/A') if isinstance(signal, dict) else getattr(signal, 'action', 'N/A')
            symbol = signal.get('symbol', '') if isinstance(signal, dict) else getattr(signal, 'symbol', '')
            lines.append(f"- {action} {symbol}: {'成功' if success else '失败'}")

        lines.extend([
            "",
            "## 请分析：",
            "1. 这次决策合理吗？(quality_score: 0-100)",
            "2. 学到了什么经验？(lessons: [])",
            "3. 下次如何改进？",
        ])

        return "\n".join(lines)

    def _parse_decision_response(self, response_text: str) -> Dict[str, Any]:
        """解析决策响应"""
        import json

        result = {
            'decision_type': 'wait',
            'signals': [],
            'analysis': response_text,
        }

        try:
            # 尝试提取 JSON
            json_start = response_text.find("```json")
            json_end = response_text.find("```", json_start + 7)

            if json_start != -1 and json_end != -1:
                json_text = response_text[json_start + 7:json_end].strip()
                data = json.loads(json_text)

                result['decision_type'] = data.get('decision_type', 'wait')
                result['signals'] = data.get('signals', [])
                result['analysis'] = response_text[:json_start].strip()

        except (json.JSONDecodeError, KeyError):
            pass

        return result

    def _parse_reflection_response(self, response_text: str) -> Dict[str, Any]:
        """解析反思响应"""
        result = {
            'reflection': response_text,
            'lessons': [],
            'quality_score': 50,
        }

        # 提取经验教训
        lines = response_text.split('\n')
        in_lessons = False
        for line in lines:
            line = line.strip()
            if '经验' in line or 'lesson' in line.lower():
                in_lessons = True
                continue
            if in_lessons and line:
                if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                    lesson = line.lstrip('-*• ').strip()
                    if lesson:
                        result['lessons'].append(lesson)
                elif line[0].isdigit() and '.' in line[:3]:
                    lesson = line.split('.', 1)[1].strip()
                    if lesson:
                        result['lessons'].append(lesson)

        # 尝试提取质量分数
        import re
        score_match = re.search(r'quality_score[:\s]*(\d+)', response_text, re.IGNORECASE)
        if score_match:
            result['quality_score'] = int(score_match.group(1))

        return result
