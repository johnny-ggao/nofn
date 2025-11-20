"""
记忆管理器

存储和检索交易案例
"""
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
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


class MemoryManager:
    """
    记忆管理器

    职责:
    1. 存储交易案例
    2. 检索相关案例
    3. 分析成功/失败模式
    4. 持久化到磁盘
    """

    def __init__(self, storage_dir: str = "data/memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.cases: List[TradingCase] = []
        self._load_from_disk()

    def add_case(self, case: TradingCase):
        """添加交易案例"""
        self.cases.append(case)
        self._save_to_disk()

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
        1. 最近表现统计
        2. 相似案例
        3. 学到的经验
        """
        lines = ["## 历史记忆", ""]

        # 最近表现
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

        # 相似案例
        if similar_cases:
            lines.append("### 相似历史案例")
            for i, case in enumerate(similar_cases[:3], 1):  # 只显示前3个
                lines.append(f"#### 案例 {i}")
                lines.append(f"- 时间: {case.timestamp.strftime('%Y-%m-%d %H:%M')}")
                if case.realized_pnl is not None:
                    result = "盈利" if case.realized_pnl > 0 else "亏损"
                    lines.append(f"- 结果: {result} ${abs(case.realized_pnl):.2f}")
                if case.lessons_learned:
                    lines.append("- 经验:")
                    for lesson in case.lessons_learned:
                        lines.append(f"  - {lesson}")
                lines.append("")

        # 总结经验教训
        all_lessons = []
        for case in recent[-10:]:  # 最近10个案例
            if case.lessons_learned:
                all_lessons.extend(case.lessons_learned)

        if all_lessons:
            lines.append("### 最近学到的经验")
            # 去重
            unique_lessons = list(set(all_lessons))
            for lesson in unique_lessons[-5:]:  # 最多显示5条
                lines.append(f"- {lesson}")
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
