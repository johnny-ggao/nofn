"""
记忆管理器

存储和检索交易案例，使用分层记忆架构：
- 短期记忆（7天）：详细案例
- 中期记忆（周）：摘要
- 长期记忆（月）：核心经验
"""
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

from numpy import floating


@dataclass
class TradingCase:
    """交易案例"""
    # 市场条件
    market_conditions: dict  # 当时的市场快照

    # 决策
    decision: dict  # 当时的决策

    # 执行结果
    execution_result: Optional[dict] = None  # 执行结果
    realized_pnl: Optional[float] = None  # 已实现盈亏

    # 反思
    reflection: Optional[str] = None  # LLM的反思
    lessons_learned: Optional[List[str]] = None  # 学到的经验

    # 元数据
    timestamp: datetime = None
    case_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.case_id is None:
            self.case_id = f"case_{int(self.timestamp.timestamp())}"

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


@dataclass
class MemorySummary:
    """记忆摘要"""
    # 时间范围
    period_start: datetime
    period_end: datetime
    period_type: str  # 'weekly' 或 'monthly'

    # 统计数据
    total_cases: int
    total_trades: int
    win_rate: float
    avg_pnl: float
    sharpe_ratio: float

    # 关键模式（LLM生成）
    key_patterns: List[str]  # 发现的交易模式
    successful_strategies: List[str]  # 成功的策略
    failed_strategies: List[str]  # 失败的策略

    # 核心经验
    lessons: List[str]  # 提炼的经验教训

    # 市场洞察
    market_insights: str  # LLM总结的市场状态

    # 元数据
    summary_id: str = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.summary_id is None:
            period_key = self.period_start.strftime("%Y%m%d")
            self.summary_id = f"{self.period_type}_{period_key}"

    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data['period_start'] = self.period_start.isoformat()
        data['period_end'] = self.period_end.isoformat()
        data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'MemorySummary':
        """从字典创建"""
        if isinstance(data['period_start'], str):
            data['period_start'] = datetime.fromisoformat(data['period_start'])
        if isinstance(data['period_end'], str):
            data['period_end'] = datetime.fromisoformat(data['period_end'])
        if isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class MemoryManager:
    """
    记忆管理器

    职责:
    1. 存储交易案例
    2. 检索相关案例
    3. 分析成功/失败模式
    4. 持久化到磁盘
    """

    def __init__(self, storage_dir: str = "data/memory", llm=None):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.cases: List[TradingCase] = []
        self.summaries: List[MemorySummary] = []
        self.llm = llm  # LLM用于生成摘要

        self._load_from_disk()
        self._load_summaries()

    def add_case(self, case: TradingCase):
        """添加交易案例"""
        self.cases.append(case)

        # 自动清理旧案例
        self._cleanup_old_cases()

        self._save_to_disk()

    def _cleanup_old_cases(self, max_cases: int = 1000, keep_days: int = 30):
        """
        清理旧案例，保留有价值的记忆

        策略：
        1. 保留最近 keep_days 天的所有案例
        2. 对于更早的案例，只保留有交易执行的
        3. 总数不超过 max_cases
        4. 被清理的案例归档到单独文件
        """
        from datetime import datetime, timedelta

        if len(self.cases) <= max_cases:
            return

        cutoff_date = datetime.now() - timedelta(days=keep_days)

        # 分类案例
        recent_cases = []      # 最近的案例（全部保留）
        valuable_old = []      # 旧的但有价值的案例
        to_archive = []        # 需要归档的案例

        for case in self.cases:
            if case.timestamp >= cutoff_date:
                # 最近的案例全部保留
                recent_cases.append(case)
            else:
                # 旧案例：只保留有交易执行的
                if case.execution_result and len(case.execution_result) > 0:
                    valuable_old.append(case)
                else:
                    to_archive.append(case)

        # 归档被清理的案例
        if to_archive:
            self._archive_cases(to_archive)

        # 合并并限制总数
        self.cases = recent_cases + valuable_old

        # 如果还是太多，按时间倒序保留前 max_cases 个
        if len(self.cases) > max_cases:
            # 超出部分也归档
            self.cases.sort(key=lambda x: x.timestamp, reverse=True)
            overflow = self.cases[max_cases:]
            if overflow:
                self._archive_cases(overflow)
            self.cases = self.cases[:max_cases]

        print(f"📊 记忆清理: 保留 {len(recent_cases)} 个最近案例 + {len(valuable_old)} 个有价值旧案例, 归档 {len(to_archive)} 个")

    def _archive_cases(self, cases: List[TradingCase]):
        """归档案例到月度文件"""
        from datetime import datetime
        import json

        archive_dir = self.storage_dir / "archives"
        archive_dir.mkdir(exist_ok=True)

        # 按月分组
        by_month = {}
        for case in cases:
            month_key = case.timestamp.strftime("%Y%m")
            if month_key not in by_month:
                by_month[month_key] = []
            by_month[month_key].append(case)

        # 保存到对应月份的归档文件
        for month_key, month_cases in by_month.items():
            archive_file = archive_dir / f"cases_{month_key}.json"

            # 加载现有归档（如果存在）
            existing = []
            if archive_file.exists():
                try:
                    with open(archive_file, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                except Exception:
                    pass

            # 合并新旧案例
            existing_ids = {c.get('case_id') for c in existing}
            new_cases = [c.to_dict() for c in month_cases if c.case_id not in existing_ids]

            if new_cases:
                all_cases = existing + new_cases
                with open(archive_file, 'w', encoding='utf-8') as f:
                    json.dump(all_cases, f, indent=2, ensure_ascii=False)

    def get_recent_cases(self, days: int = 7) -> List[TradingCase]:
        """获取最近N天的案例"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        return [case for case in self.cases if case.timestamp >= cutoff]

    def search_similar(self, market_conditions: dict, k: int = 5) -> List[TradingCase]:
        """
        检索相似案例

        简化版：基于市场趋势相似度
        未来可以使用向量嵌入提升精度
        """
        # 提取关键特征
        def extract_features(conditions: dict) -> dict:
            features = {}
            for symbol, asset in conditions.get('assets', {}).items():
                trend = asset.get('trend', {})
                features[symbol] = {
                    'trend_direction': trend.get('direction', 'neutral'),
                    'trend_strength': trend.get('strength', 50),
                }
            return features

        target_features = extract_features(market_conditions)

        # 计算相似度并排序
        def similarity_score(case: TradingCase) -> float:
            case_features = extract_features(case.market_conditions)

            score = 0.0
            for symbol in target_features:
                if symbol in case_features:
                    # 趋势方向匹配
                    if target_features[symbol]['trend_direction'] == case_features[symbol]['trend_direction']:
                        score += 50

                    # 趋势强度相似
                    strength_diff = abs(
                        target_features[symbol]['trend_strength'] - case_features[symbol]['trend_strength']
                    )
                    score += max(0, 50 - strength_diff)

            return score

        # 排序并返回top k
        scored_cases = [(case, similarity_score(case)) for case in self.cases]
        scored_cases.sort(key=lambda x: x[1], reverse=True)

        return [case for case, score in scored_cases[:k] if score > 0]

    def get_success_rate(self, conditions: Optional[dict] = None) -> float:
        """
        计算成功率

        如果提供条件，则计算该条件下的成功率
        """
        if conditions:
            matching = self.search_similar(conditions, k=20)
        else:
            matching = self.cases

        if not matching:
            return 0.5  # 默认50%

        successful = [
            case for case in matching
            if case.realized_pnl is not None and case.realized_pnl > 0
        ]

        return len(successful) / len(matching) if matching else 0.5

    def get_average_pnl(self, days: int = 7) -> float:
        """计算最近N天的平均盈亏"""
        recent = self.get_recent_cases(days)
        if not recent:
            return 0.0

        pnls = [case.realized_pnl for case in recent if case.realized_pnl is not None]
        return sum(pnls) / len(pnls) if pnls else 0.0

    def get_sharpe_ratio(self, days: int = 7) -> float | floating[Any]:
        """计算夏普比率（简化版）"""
        recent = self.get_recent_cases(days)
        if not recent:
            return 0.0

        pnls = [case.realized_pnl for case in recent if case.realized_pnl is not None]
        if not pnls:
            return 0.0

        import numpy as np
        returns = np.array(pnls)
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # 简化的夏普比率
        return mean_return / std_return

    def to_context(self, recent_days: int = 7, similar_cases: Optional[List[TradingCase]] = None) -> str:
        """
        生成记忆上下文文本（供LLM阅读）

        包括：
        1. 历史摘要（核心经验）
        2. 最近表现统计
        3. 相似案例
        """
        lines = ["## 历史记忆", ""]

        # 1. 添加历史摘要（最重要的部分）
        if self.summaries:
            lines.append("### 历史经验总结")
            lines.append("")

            # 显示最近2个摘要
            for summary in self.summaries[-2:]:
                period_name = "每周" if summary.period_type == 'weekly' else "每月"
                lines.append(f"#### {period_name}摘要 ({summary.period_start.strftime('%Y-%m-%d')} - {summary.period_end.strftime('%Y-%m-%d')})")
                lines.append(f"- 交易: {summary.total_trades} 次, 胜率 {summary.win_rate*100:.1f}%, 夏普 {summary.sharpe_ratio:.2f}")

                if summary.key_patterns:
                    lines.append("- **关键模式**:")
                    for pattern in summary.key_patterns[:3]:
                        lines.append(f"  - {pattern}")

                if summary.successful_strategies:
                    lines.append("- **成功策略**:")
                    for strategy in summary.successful_strategies[:2]:
                        lines.append(f"  - {strategy}")

                if summary.lessons:
                    lines.append("- **核心经验**:")
                    for lesson in summary.lessons[:3]:
                        lines.append(f"  - {lesson}")

                lines.append("")

        # 2. 最近表现
        recent = self.get_recent_cases(recent_days)
        if recent:
            avg_pnl = self.get_average_pnl(recent_days)
            sharpe = self.get_sharpe_ratio(recent_days)
            success_rate = self.get_success_rate()

            lines.append(f"### 最近 {recent_days} 天表现")
            lines.append(f"- 交易次数: {len(recent)}")
            lines.append(f"- 平均盈亏: ${avg_pnl:.2f}")
            lines.append(f"- 夏普比率: {sharpe:.2f}")
            lines.append(f"- 胜率: {success_rate * 100:.1f}%")
            lines.append("")

        # 3. 相似案例（减少显示，因为有摘要了）
        if similar_cases:
            lines.append("### 相似历史案例")
            for i, case in enumerate(similar_cases[:2], 1):  # 只显示前2个
                lines.append(f"#### 案例 {i}")
                lines.append(f"- 时间: {case.timestamp.strftime('%Y-%m-%d %H:%M')}")
                if case.realized_pnl is not None:
                    result = "盈利" if case.realized_pnl > 0 else "亏损"
                    lines.append(f"- 结果: {result} ${abs(case.realized_pnl):.2f}")
                if case.lessons_learned:
                    lines.append(f"- 经验: {', '.join(case.lessons_learned[:2])}")
                lines.append("")

        return "\n".join(lines)

    def _save_to_disk(self):
        """持久化到磁盘"""
        try:
            file_path = self.storage_dir / "cases.json"
            data = [case.to_dict() for case in self.cases]
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  保存记忆失败: {e}")

    def _load_from_disk(self):
        """从磁盘加载"""
        try:
            file_path = self.storage_dir / "cases.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.cases = [TradingCase.from_dict(case_data) for case_data in data]
                print(f"✅ 加载了 {len(self.cases)} 个历史案例")
        except Exception as e:
            print(f"⚠️  加载记忆失败: {e}")
            self.cases = []

    def _load_summaries(self):
        """加载摘要"""
        try:
            summary_file = self.storage_dir / "summaries.json"
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.summaries = [MemorySummary.from_dict(s) for s in data]
                print(f"✅ 加载了 {len(self.summaries)} 个记忆摘要")
        except Exception as e:
            print(f"⚠️  加载摘要失败: {e}")
            self.summaries = []

    def _save_summaries(self):
        """保存摘要"""
        try:
            summary_file = self.storage_dir / "summaries.json"
            data = [s.to_dict() for s in self.summaries]
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  保存摘要失败: {e}")

    async def generate_weekly_summary(self) -> Optional[MemorySummary]:
        """生成每周摘要"""
        if not self.llm:
            return None

        # 获取上周的案例
        now = datetime.now()
        week_start = now - timedelta(days=7)

        weekly_cases = [c for c in self.cases if week_start <= c.timestamp <= now]

        if len(weekly_cases) < 5:  # 案例太少，跳过
            return None

        # 计算统计数据
        stats = self._calculate_stats(weekly_cases)

        # LLM 生成摘要
        summary_text = await self._llm_summarize(weekly_cases, 'weekly')

        if not summary_text:
            return None

        # 创建摘要对象
        summary = MemorySummary(
            period_start=week_start,
            period_end=now,
            period_type='weekly',
            **stats,
            **self._parse_summary_text(summary_text)
        )

        # 保存并删除旧案例
        self.summaries.append(summary)
        self._save_summaries()

        # 删除已摘要的案例
        self.cases = [c for c in self.cases if c.timestamp > week_start or c.timestamp < (week_start - timedelta(days=7))]
        self._save_to_disk()

        print(f"📝 生成每周摘要: {len(weekly_cases)} 个案例 → 1 个摘要")

        return summary

    def _calculate_stats(self, cases: List[TradingCase]) -> dict:
        """计算统计数据"""
        total_cases = len(cases)

        # 只计算真实交易（开仓/平仓），不包括止盈止损修改
        real_trade_actions = {'open_long', 'open_short', 'close_position'}
        total_trades = 0

        for case in cases:
            if case.execution_result:
                for result in case.execution_result:
                    signal = result.get('signal', {})
                    # 检查是否为真实交易
                    if isinstance(signal, dict):
                        action = signal.get('action')
                    else:
                        action = getattr(signal, 'action', None)

                    if action in real_trade_actions:
                        total_trades += 1
                        break  # 每个案例只计算一次

        pnls = [c.realized_pnl for c in cases if c.realized_pnl is not None]
        wins = len([p for p in pnls if p > 0])
        win_rate = wins / len(pnls) if pnls else 0.0
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0.0

        # 计算夏普比率
        if len(pnls) >= 2:
            import numpy as np
            returns = np.array(pnls)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return {
            'total_cases': total_cases,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'sharpe_ratio': sharpe_ratio
        }

    async def _llm_summarize(self, cases: List[TradingCase], period_type: str) -> Optional[str]:
        """使用LLM生成摘要"""
        if not self.llm:
            return None

        # 准备案例数据
        cases_text = self._prepare_cases_for_summary(cases)

        prompt = f"""
请对以下{period_type}期间的交易案例进行深度分析和总结。

{cases_text}

请从以下角度提供摘要（使用JSON格式）：

1. **关键模式** (key_patterns): 列出3-5个发现的交易模式
2. **成功策略** (successful_strategies): 列出效果好的策略（2-3个）
3. **失败策略** (failed_strategies): 列出需要避免的策略（2-3个）
4. **核心经验** (lessons): 提炼3-5条最重要的经验教训
5. **市场洞察** (market_insights): 用2-3句话总结这段时间的市场状态

要求：
- 保持简洁，每条不超过50字
- 专注于可复用的模式，而非具体细节
- 提取本质规律，忽略噪音

输出格式：
```json
{{
  "key_patterns": ["模式1", "模式2", ...],
  "successful_strategies": ["策略1", "策略2", ...],
  "failed_strategies": ["策略1", "策略2", ...],
  "lessons": ["经验1", "经验2", ...],
  "market_insights": "市场洞察总结..."
}}
```
"""

        try:
            response = await self.llm.ainvoke([
                {"role": "user", "content": prompt}
            ])
            return response.content
        except Exception as e:
            print(f"⚠️  LLM摘要生成失败: {e}")
            return None

    def _prepare_cases_for_summary(self, cases: List[TradingCase]) -> str:
        """准备案例数据供LLM分析"""
        lines = []

        for i, case in enumerate(cases[:20], 1):  # 最多20个案例
            lines.append(f"### 案例 {i}")
            lines.append(f"时间: {case.timestamp.strftime('%Y-%m-%d %H:%M')}")

            # 决策
            if isinstance(case.decision, dict):
                decision_type = case.decision.get('decision_type', 'unknown')
                lines.append(f"决策: {decision_type}")
            elif isinstance(case.decision, str):
                # 截取前200字符
                lines.append(f"分析: {case.decision[:200]}...")

            # 执行结果
            if case.execution_result:
                lines.append(f"执行: {len(case.execution_result)} 个操作")

            # 盈亏
            if case.realized_pnl is not None:
                result = "盈利" if case.realized_pnl > 0 else "亏损"
                lines.append(f"结果: {result} ${abs(case.realized_pnl):.2f}")

            # 经验
            if case.lessons_learned:
                lines.append(f"经验: {', '.join(case.lessons_learned[:2])}")

            lines.append("")

        return "\n".join(lines)

    def _parse_summary_text(self, summary_text: str) -> dict:
        """解析LLM返回的摘要"""
        try:
            # 提取JSON部分
            import re
            json_match = re.search(r'```json\n(.*?)\n```', summary_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析
                json_str = summary_text

            data = json.loads(json_str)

            return {
                'key_patterns': data.get('key_patterns', []),
                'successful_strategies': data.get('successful_strategies', []),
                'failed_strategies': data.get('failed_strategies', []),
                'lessons': data.get('lessons', []),
                'market_insights': data.get('market_insights', ''),
            }
        except Exception as e:
            print(f"⚠️  解析摘要失败: {e}")
            return {
                'key_patterns': [],
                'successful_strategies': [],
                'failed_strategies': [],
                'lessons': [],
                'market_insights': summary_text[:200] if summary_text else '',
            }
