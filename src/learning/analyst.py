"""
Analyst Agent - 深度分析模块

代理式架构的核心组件：
- 自主检索 Knowledge Base 中的历史案例
- 多轮深度分析和反思
- 生成策略规则和配置调整
- 更新 Trading Agent 的行为

特性：
- 时间不敏感，可以进行深度思考
- 自主决定需要什么上下文
- 可以多轮检索和反思
- 生成可执行的规则
"""

import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path

from agno.agent import Agent
from termcolor import cprint

from .trading_knowledge import TradingKnowledge
from .agents import DynamicPromptConfig, create_model


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StrategyRule:
    """策略规则 - 由 AnalystAgent 生成的可执行规则"""

    rule_id: str
    name: str
    description: str
    condition: str  # 触发条件
    action: str  # 建议动作
    priority: int = 50  # 0-100, 越高越优先
    confidence: float = 0.5
    created_at: str = ""
    source: str = "analyst"
    active: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "StrategyRule":
        return cls(**data)


@dataclass
class AnalysisReport:
    """分析报告 - 深度分析的输出"""

    report_id: str
    timestamp: str
    period_analyzed: str  # e.g., "7d", "30d"

    # 统计摘要
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0

    # 分析结论
    patterns_found: List[str] = field(default_factory=list)
    issues_identified: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # 生成的规则
    new_rules: List[StrategyRule] = field(default_factory=list)

    # 配置调整
    config_changes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['new_rules'] = [
            r.to_dict() if isinstance(r, StrategyRule) else r
            for r in self.new_rules
        ]
        return result


# ============================================================================
# Rule Store
# ============================================================================

class RuleStore:
    """
    策略规则持久化存储

    职责：
    - 加载和保存规则到 JSON 文件
    - 管理规则的激活状态
    - 生成规则文本用于 prompt 注入
    """

    def __init__(self, rules_path: str = "data/strategy_rules.json"):
        self.rules_path = Path(rules_path)
        self.rules: Dict[str, StrategyRule] = {}
        self._load()

    # --------------------------------------------------------------------------
    # Persistence
    # --------------------------------------------------------------------------

    def _load(self) -> None:
        """从文件加载规则"""
        if not self.rules_path.exists():
            return

        try:
            data = json.loads(self.rules_path.read_text(encoding="utf-8"))
            for rule_id, rule_data in data.items():
                self.rules[rule_id] = StrategyRule.from_dict(rule_data)
            cprint(f"📋 加载 {len(self.rules)} 条策略规则", "cyan")
        except Exception as e:
            cprint(f"⚠️  加载规则失败: {e}", "yellow")

    def _save(self) -> None:
        """保存规则到文件"""
        try:
            self.rules_path.parent.mkdir(parents=True, exist_ok=True)
            data = {rule_id: rule.to_dict() for rule_id, rule in self.rules.items()}
            self.rules_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            cprint(f"⚠️  保存规则失败: {e}", "yellow")

    # --------------------------------------------------------------------------
    # CRUD Operations
    # --------------------------------------------------------------------------

    def add(self, rule: StrategyRule) -> None:
        """添加规则"""
        self.rules[rule.rule_id] = rule
        self._save()
        cprint(f"✅ 新规则: {rule.name}", "green")

    def deactivate(self, rule_id: str) -> bool:
        """停用规则"""
        if rule_id not in self.rules:
            return False
        self.rules[rule_id].active = False
        self._save()
        return True

    def get_active(self) -> List[StrategyRule]:
        """获取所有激活的规则"""
        return [r for r in self.rules.values() if r.active]

    # --------------------------------------------------------------------------
    # Prompt Generation
    # --------------------------------------------------------------------------

    def to_prompt_text(self) -> str:
        """生成规则文本用于 prompt 注入"""
        active_rules = self.get_active()
        if not active_rules:
            return ""

        lines = ["## 当前策略规则", ""]
        for rule in sorted(active_rules, key=lambda x: -x.priority):
            lines.extend([
                f"### {rule.name} (优先级: {rule.priority})",
                f"- 条件: {rule.condition}",
                f"- 动作: {rule.action}",
                f"- 置信度: {rule.confidence:.0%}",
                "",
            ])

        return "\n".join(lines)


# ============================================================================
# Analyst Agent
# ============================================================================

class AnalystAgent:
    """
    分析师 Agent - 代理式深度分析

    这是混合架构中的"慢思考"组件：
    - 时间不敏感，可以进行多轮深度分析
    - 自主从 Knowledge Base 检索历史案例
    - 识别成功/失败模式
    - 生成策略规则
    - 更新 Trading Agent 的配置
    """

    def __init__(
        self,
        knowledge: Optional[TradingKnowledge] = None,
        prompt_config: Optional[DynamicPromptConfig] = None,
        model_provider: str = "openai",
        model_id: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        rules_path: str = "data/strategy_rules.json",
        reports_path: str = "data/analysis_reports",
    ):
        self.knowledge = knowledge
        self.prompt_config = prompt_config
        self.reports_path = Path(reports_path)
        self.reports_path.mkdir(parents=True, exist_ok=True)

        # 规则存储
        self.rule_store = RuleStore(rules_path)

        # 创建模型
        self.model = create_model(
            provider=model_provider,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,  # 分析需要更稳定的输出
        )

        # 创建 Agent
        self.agent = Agent(
            name="AnalystAgent",
            model=self.model,
            instructions=[self._build_system_prompt()],
            markdown=False,
        )

        # 分析状态
        self.last_analysis_time: Optional[datetime] = None
        self.analysis_interval_hours = 24

        cprint("✅ AnalystAgent 初始化完成", "green")

    # --------------------------------------------------------------------------
    # System Prompt
    # --------------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """构建分析师系统提示词"""
        return """你是一个专业的交易系统分析师，负责深度分析交易历史并优化交易策略。

## 你的职责

1. **模式识别**：分析历史交易，识别成功和失败的模式
2. **问题诊断**：找出导致亏损的系统性问题
3. **规则生成**：提出具体可执行的策略规则
4. **配置优化**：建议调整风险参数和关注重点

## 分析框架

### 成功模式
- 什么市场条件下胜率最高？
- 哪些指标组合最有效？
- 最佳的入场时机是什么？

### 失败模式
- 常见的亏损原因是什么？
- 哪些错误在重复发生？
- 应该避免什么市场条件？

### 风险管理
- 止损设置是否合理？
- 仓位大小是否恰当？
- 盈亏比是否健康？

## 输出格式

请严格按以下 JSON 格式输出：

```json
{
    "analysis_summary": "整体分析摘要（2-3句话）",

    "patterns": {
        "successful": [
            {"pattern": "模式描述", "frequency": "出现频率", "win_rate": 0.8}
        ],
        "failed": [
            {"pattern": "模式描述", "frequency": "出现频率", "loss_rate": 0.7}
        ]
    },

    "issues": [
        {"issue": "问题描述", "severity": "high|medium|low", "suggestion": "建议"}
    ],

    "new_rules": [
        {
            "name": "规则名称",
            "condition": "触发条件（具体描述）",
            "action": "建议动作",
            "priority": 80,
            "confidence": 0.75
        }
    ],

    "config_changes": {
        "risk_level": "conservative|moderate|aggressive",
        "focus_areas": ["关注点1", "关注点2"],
        "avoid_patterns": ["避免模式1"],
        "successful_patterns": ["成功模式1"]
    },

    "performance_assessment": {
        "current_state": "improving|stable|declining",
        "key_metrics": {"win_rate": 0.6, "risk_reward": 1.5},
        "recommendations": ["建议1", "建议2"]
    }
}
```

## 重要原则

1. **基于数据**：所有结论必须基于提供的历史数据
2. **具体可执行**：规则必须足够具体，可以直接应用
3. **渐进改进**：避免一次性大幅改变，建议渐进式优化
4. **风险优先**：在收益和风险之间，优先考虑风险控制
"""

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    async def run_analysis(
        self,
        period_days: int = 7,
        account_stats: Optional[Dict] = None,
        force: bool = False,
    ) -> Optional[AnalysisReport]:
        """
        执行深度分析

        Args:
            period_days: 分析的时间范围（天）
            account_stats: 账户统计信息
            force: 强制执行分析

        Returns:
            分析报告，如果跳过分析则返回 None
        """
        if not self._should_analyze(force):
            cprint("ℹ️  距离上次分析时间不足，跳过", "yellow")
            return None

        cprint("\n" + "=" * 70, "blue")
        cprint("🔬 开始深度分析...", "blue")
        cprint("=" * 70, "blue")

        try:
            # Step 1: 检索历史案例
            cases = await self._retrieve_cases(period_days)
            if not cases:
                cprint("⚠️  没有足够的历史数据进行分析", "yellow")
                return None

            cprint(f"📚 检索到 {len(cases)} 个历史案例", "cyan")

            # Step 2: 构建分析上下文
            context = self._build_context(cases, account_stats, period_days)

            # Step 3: 执行分析
            analysis_result = await self._execute_analysis(context)

            # Step 4: 创建报告
            report = self._create_report(analysis_result, period_days)

            # Step 5: 应用改进
            self._apply_improvements(report)

            # Step 6: 保存报告
            self._save_report(report)

            self.last_analysis_time = datetime.now()

            cprint("=" * 70, "blue")
            cprint("✅ 深度分析完成", "blue")
            cprint("=" * 70, "blue")

            return report

        except Exception as e:
            cprint(f"❌ 分析失败: {e}", "red")
            import traceback
            traceback.print_exc()
            return None

    def get_rules_for_prompt(self) -> str:
        """获取规则文本用于 Trading Agent 的 prompt"""
        return self.rule_store.to_prompt_text()

    def get_stats(self) -> Dict:
        """获取分析器统计信息"""
        return {
            'total_rules': len(self.rule_store.rules),
            'active_rules': len(self.rule_store.get_active()),
            'last_analysis': (
                self.last_analysis_time.isoformat()
                if self.last_analysis_time else None
            ),
            'analysis_interval_hours': self.analysis_interval_hours,
        }

    # --------------------------------------------------------------------------
    # Analysis Pipeline
    # --------------------------------------------------------------------------

    def _should_analyze(self, force: bool = False) -> bool:
        """检查是否需要执行分析"""
        if force:
            return True
        if self.last_analysis_time is None:
            return True
        elapsed = datetime.now() - self.last_analysis_time
        return elapsed.total_seconds() > self.analysis_interval_hours * 3600

    async def _retrieve_cases(self, period_days: int) -> List[Dict]:
        """自主检索历史案例"""
        if not self.knowledge:
            return []

        cases = []
        queries = [
            "成功的交易决策 盈利",
            "失败的交易决策 亏损",
            "止损触发 风险控制",
            "趋势跟踪 入场时机",
            "高置信度决策",
        ]

        for query in queries:
            try:
                results = await self.knowledge.search(
                    query=query,
                    max_results=10,
                    min_quality=30,
                )
                cases.extend(results)
            except Exception as e:
                cprint(f"⚠️  检索失败 ({query}): {e}", "yellow")

        # 去重
        seen_ids = set()
        unique_cases = []
        for case in cases:
            case_id = case.get('name', '')
            if case_id not in seen_ids:
                seen_ids.add(case_id)
                unique_cases.append(case)

        return unique_cases

    def _build_context(
        self,
        cases: List[Dict],
        account_stats: Optional[Dict],
        period_days: int,
    ) -> str:
        """构建分析上下文"""
        lines = [
            "# 交易系统分析报告请求",
            f"分析周期: 最近 {period_days} 天",
            f"案例数量: {len(cases)}",
            "",
        ]

        # 账户统计
        if account_stats:
            lines.extend([
                "## 账户统计",
                f"- 总交易数: {account_stats.get('total_positions', 'N/A')}",
                f"- 胜率: {account_stats.get('win_rate', 0) * 100:.1f}%",
                f"- 总盈亏: ${account_stats.get('total_pnl', 0):.2f}",
                f"- 夏普比率: {account_stats.get('sharpe_ratio', 0):.2f}",
                f"- 最大回撤: ${account_stats.get('max_drawdown', 0):.2f}",
                f"- 盈亏比: {account_stats.get('risk_reward_ratio', 0):.2f}",
                "",
            ])

        # 当前配置
        if self.prompt_config:
            lines.extend([
                "## 当前策略配置",
                f"- 风险等级: {self.prompt_config.risk_level}",
                f"- 当前胜率: {self.prompt_config.win_rate * 100:.1f}%",
                f"- 绩效趋势: {self.prompt_config.recent_performance}",
                "",
            ])

            if self.prompt_config.focus_areas:
                lines.append(
                    f"- 关注重点: {', '.join(self.prompt_config.focus_areas)}"
                )
            if self.prompt_config.avoid_patterns:
                lines.append(
                    f"- 避免模式: {', '.join(self.prompt_config.avoid_patterns)}"
                )
            lines.append("")

        # 当前规则
        rules_text = self.rule_store.to_prompt_text()
        if rules_text:
            lines.extend([rules_text, ""])

        # 历史案例
        lines.extend(["## 历史交易案例", ""])

        for i, case in enumerate(cases[:20], 1):  # 限制案例数量
            content = case.get('content', '')[:800]
            meta = case.get('metadata', {})
            quality = case.get('quality_score', 'N/A')

            lines.extend([
                f"### 案例 {i}",
                f"- 时间: {meta.get('timestamp', 'N/A')}",
                f"- 标的: {meta.get('symbol', 'N/A')}",
                f"- 操作: {meta.get('action', 'N/A')}",
                f"- 成功: {meta.get('success', 'N/A')}",
                f"- 质量分: {quality}",
                "",
                content,
                "",
                "---",
                "",
            ])

        lines.append("请基于以上数据进行深度分析，并输出 JSON 格式的分析报告。")

        return "\n".join(lines)

    async def _execute_analysis(self, context: str) -> Dict:
        """执行分析（可扩展为多轮）"""
        response = await self.agent.arun(context)
        response_text = response.content if hasattr(response, 'content') else str(response)
        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> Dict:
        """解析分析响应"""
        result = {
            'analysis_summary': '',
            'patterns': {'successful': [], 'failed': []},
            'issues': [],
            'new_rules': [],
            'config_changes': {},
            'performance_assessment': {},
        }

        try:
            # 提取 JSON
            json_start = response_text.find("```json")
            json_end = response_text.find("```", json_start + 7)

            if json_start != -1 and json_end != -1:
                json_text = response_text[json_start + 7:json_end].strip()
                data = json.loads(json_text)
                result.update(data)
            else:
                # 尝试直接解析
                data = json.loads(response_text)
                result.update(data)

        except json.JSONDecodeError as e:
            cprint(f"⚠️  解析分析响应失败: {e}", "yellow")
            result['analysis_summary'] = response_text[:500]

        return result

    # --------------------------------------------------------------------------
    # Report Generation
    # --------------------------------------------------------------------------

    def _create_report(self, analysis_result: Dict, period_days: int) -> AnalysisReport:
        """创建分析报告"""
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 解析模式
        patterns_found = []
        patterns = analysis_result.get('patterns', {})
        for p in patterns.get('successful', []):
            if isinstance(p, dict):
                patterns_found.append(f"[成功] {p.get('pattern', '')}")
        for p in patterns.get('failed', []):
            if isinstance(p, dict):
                patterns_found.append(f"[失败] {p.get('pattern', '')}")

        # 解析问题
        issues = [
            i.get('issue', '')
            for i in analysis_result.get('issues', [])
            if isinstance(i, dict)
        ]

        # 解析建议
        perf = analysis_result.get('performance_assessment', {})
        recommendations = perf.get('recommendations', [])

        # 创建规则
        new_rules = []
        for rule_data in analysis_result.get('new_rules', []):
            if isinstance(rule_data, dict):
                rule = StrategyRule(
                    rule_id=f"rule_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(new_rules)}",
                    name=rule_data.get('name', 'Unnamed Rule'),
                    description=rule_data.get('condition', ''),
                    condition=rule_data.get('condition', ''),
                    action=rule_data.get('action', ''),
                    priority=rule_data.get('priority', 50),
                    confidence=rule_data.get('confidence', 0.5),
                    created_at=datetime.now().isoformat(),
                )
                new_rules.append(rule)

        # 获取统计
        metrics = perf.get('key_metrics', {})

        return AnalysisReport(
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            period_analyzed=f"{period_days}d",
            total_trades=0,
            win_rate=metrics.get('win_rate', 0),
            total_pnl=0,
            patterns_found=patterns_found,
            issues_identified=issues,
            recommendations=recommendations,
            new_rules=new_rules,
            config_changes=analysis_result.get('config_changes', {}),
        )

    def _apply_improvements(self, report: AnalysisReport) -> None:
        """应用分析报告中的改进"""
        # 添加新规则
        for rule in report.new_rules:
            self.rule_store.add(rule)

        # 更新配置
        if self.prompt_config and report.config_changes:
            self._update_config(report.config_changes)

        # 打印摘要
        self._print_summary(report)

    def _update_config(self, changes: Dict) -> None:
        """更新 prompt 配置"""
        updated = False

        if 'risk_level' in changes:
            self.prompt_config.risk_level = changes['risk_level']
            updated = True
            cprint(f"📊 风险等级调整为: {changes['risk_level']}", "yellow")

        if 'focus_areas' in changes:
            self.prompt_config.focus_areas = changes['focus_areas'][:5]
            updated = True

        if 'avoid_patterns' in changes:
            existing = set(self.prompt_config.avoid_patterns)
            new = set(changes['avoid_patterns'])
            self.prompt_config.avoid_patterns = list(existing | new)[-10:]
            updated = True

        if 'successful_patterns' in changes:
            existing = set(self.prompt_config.successful_patterns)
            new = set(changes['successful_patterns'])
            self.prompt_config.successful_patterns = list(existing | new)[-5:]
            updated = True

        if updated:
            cprint("🔄 策略配置已更新", "yellow")

    def _print_summary(self, report: AnalysisReport) -> None:
        """打印报告摘要"""
        cprint("\n📋 分析报告摘要:", "cyan")
        if report.patterns_found:
            cprint(f"   发现 {len(report.patterns_found)} 个模式", "white")
        if report.issues_identified:
            cprint(f"   识别 {len(report.issues_identified)} 个问题", "white")
        if report.new_rules:
            cprint(f"   生成 {len(report.new_rules)} 条新规则", "white")
        if report.recommendations:
            cprint(f"   提出 {len(report.recommendations)} 条建议", "white")

    def _save_report(self, report: AnalysisReport) -> None:
        """保存分析报告"""
        try:
            report_file = self.reports_path / f"{report.report_id}.json"
            report_file.write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            cprint(f"💾 报告已保存: {report_file}", "green")
        except Exception as e:
            cprint(f"⚠️  保存报告失败: {e}", "yellow")
