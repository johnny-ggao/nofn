"""
Trading Agents - 交易决策和评估 Agent

混合架构核心组件:
- DecisionAgent: 交易决策 (动态 prompt)
- EvaluationAgent: 即时评估 (小幅调整 prompt)

数据流:
DecisionAgent ──▶ Knowledge Base ──▶ AnalystAgent (见 analyst.py)
      ▲                                    │
      └──── PromptConfig ◀── 规则/配置 ◀──┘
"""

import json
import re
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from termcolor import cprint

from ..engine.market_snapshot import MarketSnapshot
from .trading_knowledge import TradingKnowledge, TradingCase


# ============================================================================
# Model Factory
# ============================================================================

def create_model(
    provider: str = "openai",
    model_id: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
):
    """创建 Agno 模型实例"""
    if provider == "anthropic":
        return Claude(id=model_id, api_key=api_key)

    # OpenAI 兼容模型
    role_map = None
    if base_url and ("dashscope" in base_url or "aliyun" in base_url):
        role_map = {
            "system": "system",
            "developer": "system",
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


# ============================================================================
# Dynamic Prompt Configuration
# ============================================================================

@dataclass
class DynamicPromptConfig:
    """
    动态 Prompt 配置

    用于存储和管理 DecisionAgent 的动态调整参数。
    配置会被持久化到 JSON 文件，并在启动时加载。
    """
    # 基础配置
    base_prompt: str = ""

    # 风险管理
    risk_level: str = "moderate"  # conservative, moderate, aggressive

    # 策略调整
    focus_areas: List[str] = field(default_factory=list)
    avoid_patterns: List[str] = field(default_factory=list)
    successful_patterns: List[str] = field(default_factory=list)
    recent_lessons: List[str] = field(default_factory=list)

    # 绩效指标
    win_rate: float = 0.5
    avg_pnl: float = 0.0
    recent_performance: str = "neutral"  # improving, declining, neutral

    # 元数据
    last_updated: str = ""

    def to_prompt_section(self) -> str:
        """生成动态 prompt 片段"""
        sections = []

        # 风险策略
        risk_guidance = {
            "conservative": "保守交易，优先保护本金，只在高确定性机会入场，严格止损",
            "moderate": "平衡风险收益，寻找合理的风险回报比，适度仓位",
            "aggressive": "积极交易，可以承受更大波动，抓住更多机会",
        }
        sections.append(f"## 当前风险策略\n{risk_guidance.get(self.risk_level, risk_guidance['moderate'])}")

        # 关注重点
        if self.focus_areas:
            items = "\n".join(f"- {area}" for area in self.focus_areas[:5])
            sections.append(f"## 重点关注\n{items}")

        # 应避免的模式
        if self.avoid_patterns:
            items = "\n".join(f"- {p}" for p in self.avoid_patterns[:5])
            sections.append(f"## 应避免的错误\n{items}")

        # 成功模式
        if self.successful_patterns:
            items = "\n".join(f"- {p}" for p in self.successful_patterns[:3])
            sections.append(f"## 有效的策略\n{items}")

        # 近期教训
        if self.recent_lessons:
            items = "\n".join(f"- {lesson}" for lesson in self.recent_lessons[:5])
            sections.append(f"## 近期经验\n{items}")

        # 绩效状态
        perf_text = {
            "improving": "近期表现良好，保持当前策略",
            "declining": "近期表现下滑，需要更谨慎",
            "neutral": "表现平稳，寻找突破机会",
        }
        sections.append(f"## 绩效状态\n{perf_text.get(self.recent_performance, perf_text['neutral'])}")
        sections.append(f"当前胜率: {self.win_rate * 100:.1f}%")

        return "\n\n".join(sections)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "DynamicPromptConfig":
        return cls(**data)


# ============================================================================
# Trading Agents
# ============================================================================

class TradingAgents:
    """
    交易 Agents 管理器

    职责:
    - 管理 DecisionAgent 和 EvaluationAgent
    - 维护动态 prompt 配置
    - 管理知识库连接
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
        knowledge_db_path: str = "data/trading_knowledge",
        knowledge_enabled: bool = True,
        embedder_model: str = "text-embedding-3-small",
        embedder_api_key: Optional[str] = None,
        embedder_base_url: Optional[str] = None,
        prompt_config_path: str = "data/prompt_config.json",
    ):
        self.db = db
        self.prompt_config_path = prompt_config_path

        # 创建模型
        self.model = create_model(
            provider=model_provider,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

        # 加载基础系统提示词
        self.base_system_prompt = self._load_system_prompt(system_prompt_path)

        # 加载动态配置
        self.prompt_config = self._load_prompt_config()

        # 初始化知识库
        self.knowledge: Optional[TradingKnowledge] = None
        self.knowledge_enabled = knowledge_enabled
        if knowledge_enabled:
            self._init_knowledge(
                knowledge_db_path, embedder_model,
                embedder_api_key or api_key,
                embedder_base_url or base_url,
            )

        # 决策历史
        self.decision_history: List[Dict] = []
        self._max_history = 50

        # 创建 Agents
        self._init_agents()

    def _load_system_prompt(self, path: Optional[str]) -> str:
        """加载系统提示词"""
        if path:
            return Path(path).read_text(encoding="utf-8")
        return self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是一个专业的加密货币交易分析师。

职责：
1. 分析市场数据和技术指标
2. 结合历史经验做出交易决策
3. 提供清晰的买入/卖出/持有建议

核心原则：
- 风险控制优先，每笔交易必须设置止损
- 趋势跟踪，顺势而为
- 仓位管理，不过度交易
- 保持客观，避免情绪化决策
"""

    def _load_prompt_config(self) -> DynamicPromptConfig:
        """加载动态配置"""
        path = Path(self.prompt_config_path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                config = DynamicPromptConfig.from_dict(data)
                cprint(f"📋 已加载 prompt 配置 (更新: {config.last_updated[:10]})", "cyan")
                return config
            except Exception as e:
                cprint(f"⚠️  加载配置失败: {e}", "yellow")

        return DynamicPromptConfig(
            base_prompt=self.base_system_prompt,
            last_updated=datetime.now().isoformat(),
        )

    def _save_prompt_config(self) -> None:
        """保存动态配置"""
        try:
            path = Path(self.prompt_config_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.prompt_config.last_updated = datetime.now().isoformat()
            path.write_text(
                json.dumps(self.prompt_config.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            cprint(f"⚠️  保存配置失败: {e}", "yellow")

    def _init_knowledge(
        self,
        db_path: str,
        embedder_model: str,
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> None:
        """初始化知识库"""
        try:
            self.knowledge = TradingKnowledge(
                db_path=db_path,
                embedder_model=embedder_model,
                embedder_api_key=api_key,
                embedder_base_url=base_url,
            )
        except Exception as e:
            cprint(f"⚠️  知识库初始化失败: {e}", "yellow")
            self.knowledge = None

    def _init_agents(self) -> None:
        """初始化 Agents"""
        # Decision Agent
        self.decision_agent = Agent(
            name="DecisionAgent",
            model=self.model,
            db=self.db,
            session_id="decision",
            instructions=[self._build_decision_prompt()],
            add_history_to_context=True,
            num_history_runs=3,
            enable_session_summaries=True,
            markdown=True,
        )

        # Evaluation Agent
        self.evaluation_agent = Agent(
            name="EvaluationAgent",
            model=self.model,
            db=self.db,
            session_id="evaluation",
            instructions=[self._evaluation_system_prompt()],
            add_history_to_context=True,
            num_history_runs=10,
            markdown=False,
        )

        cprint("✅ TradingAgents 初始化完成", "green")

    def _build_decision_prompt(self) -> str:
        """构建决策 prompt"""
        return "\n".join([
            self.base_system_prompt,
            "",
            "=" * 50,
            "# 动态策略调整 (基于历史表现)",
            "=" * 50,
            "",
            self.prompt_config.to_prompt_section(),
        ])

    def _evaluation_system_prompt(self) -> str:
        """评估系统提示词"""
        return """你是交易系统的元评估模块，负责分析决策并优化策略。

职责：
1. 分析交易决策和执行结果
2. 识别成功和失败的模式
3. 提出策略调整建议

输出格式 (JSON):
```json
{
    "analysis": "整体分析",
    "quality_score": 0-100,
    "lessons": ["经验1", "经验2"],
    "prompt_adjustments": {
        "risk_level": "conservative | moderate | aggressive",
        "focus_areas": ["关注点"],
        "avoid_patterns": ["避免模式"],
        "successful_patterns": ["成功模式"],
        "recent_lessons": ["经验教训"]
    },
    "performance_trend": "improving | declining | neutral"
}
```

原则：
- 基于数据分析，不主观臆断
- 建议具体可执行
- 保持连续性，避免频繁大幅调整
"""

    def update_decision_agent_prompt(self) -> None:
        """更新决策 Agent 的 prompt"""
        self.decision_agent = Agent(
            name="DecisionAgent",
            model=self.model,
            db=self.db,
            session_id="decision",
            instructions=[self._build_decision_prompt()],
            add_history_to_context=True,
            num_history_runs=3,
            enable_session_summaries=True,
            markdown=True,
        )
        cprint("🔄 DecisionAgent prompt 已更新", "cyan")

    # ========================================================================
    # Core Methods
    # ========================================================================

    async def make_decision(
        self,
        market_snapshot: MarketSnapshot,
        memory_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行交易决策"""
        # 检索知识库
        knowledge_context = await self._retrieve_knowledge(market_snapshot)

        # 合并上下文
        combined_context = self._merge_context(memory_context, knowledge_context)

        # 构建请求
        request = self._build_decision_request(market_snapshot, combined_context)

        # 调用 Agent
        response = await self.decision_agent.arun(request)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # 解析响应
        decision = self._parse_decision(response_text)

        # 记录历史
        self._record_decision(market_snapshot, decision)

        return decision

    async def evaluate_and_adjust(
        self,
        decision: Dict[str, Any],
        execution_results: List[Dict],
        account_info: Optional[Dict] = None,
        market_snapshot: Optional[MarketSnapshot] = None,
    ) -> Dict[str, Any]:
        """评估决策并调整策略"""
        # 构建评估请求
        request = self._build_evaluation_request(
            decision, execution_results, account_info, market_snapshot
        )

        # 调用 Agent
        response = await self.evaluation_agent.arun(request)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # 解析评估结果
        evaluation = self._parse_evaluation(response_text)

        # 更新历史
        self._update_decision_history(decision, execution_results, evaluation)

        # 应用调整
        prompt_updated = self._apply_adjustments(evaluation, account_info)
        evaluation['prompt_updated'] = prompt_updated

        return evaluation

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _retrieve_knowledge(self, snapshot: MarketSnapshot) -> str:
        """从知识库检索相关历史"""
        if not self.knowledge or not self.knowledge_enabled:
            return ""

        try:
            market_summary = snapshot.to_text()[:500]
            symbol = list(snapshot.assets.keys())[0] if snapshot.assets else "BTC"

            context = await self.knowledge.get_relevant_context(
                market_summary=market_summary,
                symbol=symbol,
                max_cases=3,
            )

            if context:
                count = context.count('### 案例')
                cprint(f"📚 检索到 {count} 条相关历史", "cyan")

            return context
        except Exception as e:
            cprint(f"⚠️  知识库检索失败: {e}", "yellow")
            return ""

    def _merge_context(self, memory: Optional[str], knowledge: str) -> Optional[str]:
        """合并上下文"""
        parts = [p for p in [memory, knowledge] if p]
        return "\n\n".join(parts) if parts else None

    def _build_decision_request(
        self,
        snapshot: MarketSnapshot,
        context: Optional[str],
    ) -> str:
        """构建决策请求"""
        lines = [
            "请分析以下市场情况并做出交易决策：",
            "",
            "=" * 60,
            snapshot.to_text(),
            "=" * 60,
        ]

        if context:
            lines.extend(["", "## 历史记忆", context, "=" * 60])

        lines.extend([
            "",
            "## 决策输出格式",
            "```json",
            '{',
            '  "decision_type": "trade | hold | wait",',
            '  "signals": [{',
            '    "action": "open_long | open_short | close_position | hold | wait",',
            '    "symbol": "BTC/USDC:USDC",',
            '    "amount": 0.001,',
            '    "leverage": 3,',
            '    "stop_loss": 88000.0,',
            '    "take_profit": 96000.0,',
            '    "confidence": 85,',
            '    "reason": "原因说明"',
            '  }]',
            '}',
            "```",
        ])

        return "\n".join(lines)

    def _build_evaluation_request(
        self,
        decision: Dict[str, Any],
        execution_results: List[Dict],
        account_info: Optional[Dict],
        market_snapshot: Optional[MarketSnapshot],
    ) -> str:
        """构建评估请求"""
        lines = ["请评估以下交易过程：", ""]

        # 账户状态
        lines.append("## 账户状态")
        if account_info:
            balance = account_info.get('balance', {})
            stats = account_info.get('statistics', {})
            lines.extend([
                f"- 余额: ${balance.get('total', 0):.2f}",
                f"- 胜率: {stats.get('win_rate', 0) * 100:.1f}%",
                f"- 总盈亏: ${stats.get('total_pnl', 0):.2f}",
                f"- 夏普比率: {stats.get('sharpe_ratio', 0):.2f}",
            ])
        else:
            lines.append("（账户信息不可用）")

        # 当前配置
        lines.extend([
            "",
            "## 当前策略",
            f"- 风险等级: {self.prompt_config.risk_level}",
            f"- 胜率: {self.prompt_config.win_rate * 100:.1f}%",
        ])

        # 市场状态
        if market_snapshot:
            lines.extend(["", "## 市场状态", market_snapshot.to_text()[:800]])

        # 决策内容
        lines.extend(["", "## 本次决策", decision.get('analysis', 'N/A')[:400]])

        # 执行结果
        lines.extend(["", "## 执行结果"])
        for r in execution_results:
            signal = r.get('signal', {})
            result = r.get('result', {})
            action = signal.get('action', 'N/A') if isinstance(signal, dict) else 'N/A'
            symbol = signal.get('symbol', '') if isinstance(signal, dict) else ''
            success = "成功" if result.get('success') else "失败"
            pnl = result.get('pnl', 0) or 0
            lines.append(f"- {action} {symbol}: {success} (PnL: ${pnl:.2f})")

        # 历史记录
        if self.decision_history:
            lines.extend(["", "## 近期记录"])
            for rec in self.decision_history[-5:]:
                ts = rec.get('timestamp', '')[:10]
                dec_type = rec.get('decision', {}).get('decision_type', 'N/A')
                score = rec.get('evaluation', {}).get('quality_score', 'N/A')
                lines.append(f"- [{ts}] {dec_type} | 质量: {score}")

        lines.extend(["", "## 请输出 JSON 格式的评估结果"])

        return "\n".join(lines)

    def _parse_decision(self, text: str) -> Dict[str, Any]:
        """解析决策响应"""
        result = {'decision_type': 'wait', 'signals': [], 'analysis': text}

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                result['decision_type'] = data.get('decision_type', 'wait')
                result['signals'] = data.get('signals', [])
                result['analysis'] = text[:text.find('```json')].strip()
        except (json.JSONDecodeError, AttributeError):
            pass

        return result

    def _parse_evaluation(self, text: str) -> Dict[str, Any]:
        """解析评估响应"""
        result = {
            'analysis': text,
            'quality_score': 50,
            'lessons': [],
            'prompt_adjustments': {},
            'performance_trend': 'neutral',
        }

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                result.update({
                    'analysis': data.get('analysis', result['analysis']),
                    'quality_score': data.get('quality_score', 50),
                    'lessons': data.get('lessons', []),
                    'prompt_adjustments': data.get('prompt_adjustments', {}),
                    'performance_trend': data.get('performance_trend', 'neutral'),
                })
        except (json.JSONDecodeError, AttributeError) as e:
            cprint(f"⚠️  解析评估失败: {e}", "yellow")
            # 尝试提取质量分数
            score_match = re.search(r'quality_score[:\s]*(\d+)', text, re.IGNORECASE)
            if score_match:
                result['quality_score'] = int(score_match.group(1))

        return result

    def _record_decision(self, snapshot: MarketSnapshot, decision: Dict) -> None:
        """记录决策历史"""
        self.decision_history.append({
            "timestamp": datetime.now().isoformat(),
            "market_summary": snapshot.to_text()[:500],
            "decision": decision,
        })
        # 保持历史记录数量限制
        if len(self.decision_history) > self._max_history:
            self.decision_history = self.decision_history[-self._max_history:]

    def _update_decision_history(
        self,
        decision: Dict,
        execution_results: List[Dict],
        evaluation: Dict,
    ) -> None:
        """更新决策历史的执行结果"""
        for record in reversed(self.decision_history):
            if record.get('decision') == decision:
                record['execution_results'] = execution_results
                record['evaluation'] = evaluation
                break

    def _apply_adjustments(
        self,
        evaluation: Dict,
        account_info: Optional[Dict],
    ) -> bool:
        """应用 prompt 调整"""
        adjustments = evaluation.get('prompt_adjustments', {})
        if not adjustments:
            return False

        updated = False
        config = self.prompt_config

        # 风险等级
        if 'risk_level' in adjustments:
            level = adjustments['risk_level']
            if level in ['conservative', 'moderate', 'aggressive'] and config.risk_level != level:
                config.risk_level = level
                updated = True
                cprint(f"📊 风险等级: {level}", "yellow")

        # 关注重点
        if adjustments.get('focus_areas'):
            config.focus_areas = adjustments['focus_areas'][:5]
            updated = True

        # 避免模式
        if adjustments.get('avoid_patterns'):
            existing = set(config.avoid_patterns)
            new = set(adjustments['avoid_patterns'])
            config.avoid_patterns = list(existing | new)[-10:]
            updated = True

        # 成功模式
        if adjustments.get('successful_patterns'):
            existing = set(config.successful_patterns)
            new = set(adjustments['successful_patterns'])
            config.successful_patterns = list(existing | new)[-5:]
            updated = True

        # 近期教训
        if adjustments.get('recent_lessons'):
            config.recent_lessons = adjustments['recent_lessons'][:5]
            updated = True

        # 绩效趋势
        if 'performance_trend' in evaluation:
            config.recent_performance = evaluation['performance_trend']

        # 统计数据
        if account_info:
            stats = account_info.get('statistics', {})
            config.win_rate = stats.get('win_rate', 0.5)
            config.avg_pnl = stats.get('avg_pnl', 0.0)

        if updated:
            self._save_prompt_config()
            self.update_decision_agent_prompt()

        return updated

    # ========================================================================
    # Knowledge Base Methods
    # ========================================================================

    async def save_case_to_knowledge(
        self,
        case_id: str,
        symbol: str,
        market_snapshot: MarketSnapshot,
        decision: Dict[str, Any],
        execution_results: List[Dict],
        reflection: Optional[Dict[str, Any]] = None,
    ) -> None:
        """保存案例到知识库"""
        if not self.knowledge or not self.knowledge_enabled:
            return

        try:
            signals = decision.get('signals', [])
            first_signal = signals[0] if signals else {}

            success = all(
                r.get('result', {}).get('success', False)
                for r in execution_results
            ) if execution_results else False

            total_pnl = sum(
                r.get('result', {}).get('pnl', 0) or 0
                for r in execution_results
            )

            case = TradingCase(
                case_id=case_id,
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                market_summary=market_snapshot.to_text()[:1000],
                decision_type=decision.get('decision_type', 'wait'),
                action=first_signal.get('action', 'wait') if isinstance(first_signal, dict) else 'wait',
                confidence=first_signal.get('confidence', 50) if isinstance(first_signal, dict) else 50,
                reason=first_signal.get('reason', '') if isinstance(first_signal, dict) else '',
                success=success,
                pnl=total_pnl if total_pnl != 0 else None,
                reflection=reflection.get('reflection', '') if reflection else None,
                lessons=reflection.get('lessons', []) if reflection else [],
                quality_score=reflection.get('quality_score', 50) if reflection else 50,
            )

            await self.knowledge.add_case(case)

        except Exception as e:
            cprint(f"⚠️  保存案例失败: {e}", "yellow")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_prompt_config(self) -> Dict:
        """获取当前配置"""
        return self.prompt_config.to_dict()

    def get_decision_history_summary(self) -> Dict:
        """获取决策历史摘要"""
        if not self.decision_history:
            return {"total": 0, "recent": []}

        recent = [
            {
                "timestamp": r.get('timestamp', ''),
                "decision_type": r.get('decision', {}).get('decision_type', 'N/A'),
                "quality_score": r.get('evaluation', {}).get('quality_score', 'N/A'),
            }
            for r in self.decision_history[-10:]
        ]

        return {"total": len(self.decision_history), "recent": recent}

    # ========================================================================
    # Backward Compatibility
    # ========================================================================

    async def reflect(
        self,
        decision: Dict[str, Any],
        execution_results: List[Dict],
        account_info: Optional[Dict] = None,
        market_snapshot: Optional[MarketSnapshot] = None,
    ) -> Dict[str, Any]:
        """向后兼容: 反思方法"""
        evaluation = await self.evaluate_and_adjust(
            decision, execution_results, account_info, market_snapshot
        )
        return {
            'reflection': evaluation.get('analysis', ''),
            'lessons': evaluation.get('lessons', []),
            'quality_score': evaluation.get('quality_score', 50),
        }

    async def generate_summary(
        self,
        cases_text: str,
        account_info: Optional[Dict] = None,
    ) -> str:
        """向后兼容: 生成摘要"""
        return f"当前策略配置:\n{self.prompt_config.to_prompt_section()}"
