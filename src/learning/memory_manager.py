"""
记忆管理器（SQLite 版本）

存储和检索交易案例，使用分层记忆架构：
- 短期记忆（7天）：详细案例
- 中期记忆（周）：摘要
- 长期记忆（月）：核心经验

使用 SQLite 数据库进行持久化存储
"""
import sqlite3
import json
import time
import random
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading

from numpy import floating


@dataclass
class TradingCase:
    """交易案例"""
    # 市场条件
    market_conditions: dict  # 当时的市场快照

    # 决策
    decision: str  # 当时的决策（LLM分析文本）

    # 执行结果
    execution_result: Optional[List[dict]] = None  # 执行结果
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
    记忆管理器（使用 SQLite）

    职责:
    1. 存储交易案例
    2. 检索相关案例
    3. 分析成功/失败模式
    4. 持久化到 SQLite 数据库
    5. 生成和存储记忆摘要
    """

    def __init__(self, db_path: str = "data/memory.db", llm=None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self.llm = llm  # LLM用于生成摘要

        # 初始化数据库
        self._init_database()

        # 加载统计信息
        stats = self._get_db_stats()
        print(f"✅ 加载了 {stats['cases_count']} 个历史案例, {stats['summaries_count']} 个摘要")

    def _get_connection(self) -> sqlite3.Connection:
        """获取线程本地的数据库连接"""
        if not hasattr(self._local, 'conn'):
            # 增加 timeout 到 30 秒以处理并发访问
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row

            # 启用 WAL 模式以支持并发读写
            self._local.conn.execute('PRAGMA journal_mode=WAL')
            self._local.conn.execute('PRAGMA busy_timeout=30000')  # 30秒

        return self._local.conn

    def _init_database(self):
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 创建交易案例表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_cases (
                case_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                market_conditions TEXT NOT NULL,
                decision TEXT NOT NULL,
                execution_result TEXT,
                realized_pnl REAL,
                reflection TEXT,
                lessons_learned TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cases_timestamp ON trading_cases(timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cases_realized_pnl ON trading_cases(realized_pnl)')

        # 创建记忆摘要表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_summaries (
                summary_id TEXT PRIMARY KEY,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                period_type TEXT NOT NULL,
                total_cases INTEGER NOT NULL,
                total_trades INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                avg_pnl REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                key_patterns TEXT NOT NULL,
                successful_strategies TEXT NOT NULL,
                failed_strategies TEXT NOT NULL,
                lessons TEXT NOT NULL,
                market_insights TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_summaries_period ON memory_summaries(period_start DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_summaries_type ON memory_summaries(period_type)')

        conn.commit()

    def _get_db_stats(self) -> dict:
        """获取数据库统计信息"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM trading_cases')
        cases_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM memory_summaries')
        summaries_count = cursor.fetchone()[0]

        return {
            'cases_count': cases_count,
            'summaries_count': summaries_count
        }

    def add_case(self, case: TradingCase):
        """添加交易案例（带强化重试机制）"""
        max_retries = 10  # 增加到10次
        base_delay = 0.5  # 基础等待500ms

        for attempt in range(max_retries):
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO trading_cases
                    (case_id, timestamp, market_conditions, decision, execution_result,
                     realized_pnl, reflection, lessons_learned)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    case.case_id,
                    case.timestamp.isoformat(),
                    json.dumps(case.market_conditions, ensure_ascii=False),
                    case.decision,
                    json.dumps(case.execution_result, ensure_ascii=False) if case.execution_result else None,
                    float(case.realized_pnl) if case.realized_pnl is not None else None,
                    case.reflection,
                    json.dumps(case.lessons_learned, ensure_ascii=False) if case.lessons_learned else None,
                ))

                conn.commit()

                # 自动清理旧案例（保留最近1000个）
                self._cleanup_old_cases(max_cases=1000, keep_days=30)

                # 成功，退出重试循环
                if attempt > 0:
                    print(f"✅ 案例保存成功（重试 {attempt} 次后）")
                break

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    # 计算延迟：基础延迟 + 指数退避 + 随机抖动
                    exponential_delay = base_delay * (1.5 ** attempt)
                    jitter = random.uniform(0, 0.3)  # 随机抖动0-300ms
                    retry_delay = exponential_delay + jitter

                    print(f"⚠️  数据库锁定，{retry_delay:.2f}秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    # 最后一次重试失败或其他错误，抛出异常
                    print(f"❌ 案例保存失败: 重试 {max_retries} 次后仍然数据库锁定")
                    raise

    def _cleanup_old_cases(self, max_cases: int = 1000, keep_days: int = 30):
        """
        清理旧案例

        策略：
        1. 保留最近 keep_days 天的所有案例
        2. 总数不超过 max_cases
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # 统计总数
        cursor.execute('SELECT COUNT(*) FROM trading_cases')
        total_count = cursor.fetchone()[0]

        if total_count <= max_cases:
            return

        # 保留最近的 max_cases 个案例
        cutoff_date = (datetime.now() - timedelta(days=keep_days)).isoformat()

        cursor.execute('''
            DELETE FROM trading_cases
            WHERE case_id IN (
                SELECT case_id FROM trading_cases
                WHERE timestamp < ?
                ORDER BY timestamp DESC
                LIMIT -1 OFFSET ?
            )
        ''', (cutoff_date, max_cases))

        deleted = cursor.rowcount
        if deleted > 0:
            print(f"📊 记忆清理: 删除了 {deleted} 个旧案例, 保留最近 {max_cases} 个")

        conn.commit()

    def get_recent_cases(self, days: int = 7, limit: Optional[int] = None) -> List[TradingCase]:
        """获取最近N天的案例"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        query = '''
            SELECT * FROM trading_cases
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        '''

        if limit:
            query += f' LIMIT {limit}'

        cursor.execute(query, (cutoff,))

        return [self._row_to_case(row) for row in cursor.fetchall()]

    def search_similar(self, market_conditions: dict, k: int = 5) -> List[TradingCase]:
        """
        检索相似案例

        简化版：基于市场趋势相似度
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

        # 获取所有案例
        cases = self.get_recent_cases(days=30)  # 只搜索最近30天

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
        scored_cases = [(case, similarity_score(case)) for case in cases]
        scored_cases.sort(key=lambda x: x[1], reverse=True)

        return [case for case, score in scored_cases[:k] if score > 0]

    def get_success_rate(self, conditions: Optional[dict] = None) -> float:
        """计算成功率"""
        if conditions:
            matching = self.search_similar(conditions, k=20)
        else:
            matching = self.get_recent_cases(days=30)

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
        summaries = self._get_recent_summaries(limit=2)
        if summaries:
            lines.append("### 历史经验总结")
            lines.append("")

            for summary in summaries:
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

    def _get_recent_summaries(self, limit: int = 2) -> List[MemorySummary]:
        """获取最近的摘要"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM memory_summaries
            ORDER BY period_start DESC
            LIMIT ?
        ''', (limit,))

        return [self._row_to_summary(row) for row in cursor.fetchall()]

    async def generate_weekly_summary(self, account_info: dict = None) -> Optional[MemorySummary]:
        """生成每周摘要（可包含账户信息）"""
        if not self.llm:
            return None

        # 获取上周的案例
        now = datetime.now()
        week_start = now - timedelta(days=7)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM trading_cases
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
        ''', (week_start.isoformat(), now.isoformat()))

        weekly_cases = [self._row_to_case(row) for row in cursor.fetchall()]

        if len(weekly_cases) < 5:  # 案例太少，跳过
            return None

        # 计算统计数据
        stats = self._calculate_stats(weekly_cases)

        # LLM 生成摘要（包含账户信息）
        summary_text = await self._llm_summarize(weekly_cases, 'weekly', account_info)

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

        # 保存摘要
        self._save_summary(summary)

        print(f"📝 生成每周摘要: {len(weekly_cases)} 个案例 → 1 个摘要")

        return summary

    def _save_summary(self, summary: MemorySummary):
        """保存摘要到数据库"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO memory_summaries
            (summary_id, period_start, period_end, period_type, total_cases, total_trades,
             win_rate, avg_pnl, sharpe_ratio, key_patterns, successful_strategies,
             failed_strategies, lessons, market_insights, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            summary.summary_id,
            summary.period_start.isoformat(),
            summary.period_end.isoformat(),
            summary.period_type,
            summary.total_cases,
            summary.total_trades,
            summary.win_rate,
            summary.avg_pnl,
            summary.sharpe_ratio,
            json.dumps(summary.key_patterns, ensure_ascii=False),
            json.dumps(summary.successful_strategies, ensure_ascii=False),
            json.dumps(summary.failed_strategies, ensure_ascii=False),
            json.dumps(summary.lessons, ensure_ascii=False),
            summary.market_insights,
            summary.created_at.isoformat(),
        ))

        conn.commit()

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

    async def _llm_summarize(self, cases: List[TradingCase], period_type: str, account_info: dict = None) -> Optional[str]:
        """使用LLM生成摘要（可包含账户信息）"""
        if not self.llm:
            return None

        # 准备案例数据
        cases_text = self._prepare_cases_for_summary(cases)

        # 准备账户信息文本
        account_text = ""
        if account_info:
            balance = account_info.get('balance', {})
            stats = account_info.get('statistics', {})
            positions = account_info.get('open_positions', [])

            account_text = f"""
## 当前账户状态

- 账户余额: ${balance.get('total', 0):.2f}
- 可用资金: ${balance.get('available', 0):.2f}
- 冻结保证金: ${balance.get('frozen', 0):.2f}
"""
            if stats:
                account_text += f"""- 累计平仓次数: {stats.get('total_positions', 0)} 次
  - 盈利次数: {stats.get('win_count', 0)} 次
  - 亏损次数: {stats.get('loss_count', 0)} 次
  - 胜率: {stats.get('win_rate', 0) * 100:.1f}%
- 已实现总盈亏: ${stats.get('total_pnl', 0):.2f}
  - 最大盈利: ${stats.get('max_profit', 0):.2f}
  - 最大亏损: ${stats.get('max_loss', 0):.2f}
"""
            if positions:
                unrealized_total = sum(float(p.get('unrealized_pnl', 0)) for p in positions)
                account_text += f"""- 当前持仓: {len(positions)} 个
- 未实现盈亏: ${unrealized_total:.2f}
"""

        prompt = f"""
请对以下{period_type}期间的交易案例进行深度分析和总结。

{account_text}

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
- 如果提供了账户信息，请结合账户整体表现进行分析

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
            if isinstance(case.decision, str):
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

    def _row_to_case(self, row: sqlite3.Row) -> TradingCase:
        """将数据库行转换为 TradingCase"""
        return TradingCase(
            case_id=row['case_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            market_conditions=json.loads(row['market_conditions']),
            decision=row['decision'],
            execution_result=json.loads(row['execution_result']) if row['execution_result'] else None,
            realized_pnl=row['realized_pnl'],
            reflection=row['reflection'],
            lessons_learned=json.loads(row['lessons_learned']) if row['lessons_learned'] else None,
        )

    def _row_to_summary(self, row: sqlite3.Row) -> MemorySummary:
        """将数据库行转换为 MemorySummary"""
        return MemorySummary(
            summary_id=row['summary_id'],
            period_start=datetime.fromisoformat(row['period_start']),
            period_end=datetime.fromisoformat(row['period_end']),
            period_type=row['period_type'],
            total_cases=row['total_cases'],
            total_trades=row['total_trades'],
            win_rate=row['win_rate'],
            avg_pnl=row['avg_pnl'],
            sharpe_ratio=row['sharpe_ratio'],
            key_patterns=json.loads(row['key_patterns']),
            successful_strategies=json.loads(row['successful_strategies']),
            failed_strategies=json.loads(row['failed_strategies']),
            lessons=json.loads(row['lessons']),
            market_insights=row['market_insights'],
            created_at=datetime.fromisoformat(row['created_at']),
        )

    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            delattr(self._local, 'conn')

    def __del__(self):
        """析构时关闭连接"""
        self.close()
