"""
Reflector - ACE 反思者模块

职责：
1. 分析执行结果
2. 诊断失败原因
3. 评估策略有效性
4. 提取新洞察
"""

import json
from termcolor import cprint

from ..models import Reflection, ExecutionTrace, StrategyEvaluation, FailureType
from ..utils import PromptBuilder


class Reflector:
    """ACE 反思者"""

    def __init__(self, llm):
        self.llm = llm

    async def reflect(self, trace: ExecutionTrace) -> Reflection:
        """对执行轨迹进行反思"""
        reflection = Reflection()
        reflection.trace_id = trace.trace_id

        try:
            # 1. 构建反思 Prompt
            prompt = PromptBuilder.build_reflector_prompt(trace)

            # 2. LLM 分析
            cprint("🤔 Reflector 正在分析...", "magenta")
            response = await self.llm.ainvoke([
                {"role": "system", "content": "你是一个交易分析专家，负责反思交易决策的质量。"},
                {"role": "user", "content": prompt}
            ])

            reflection.reflection_text = response.content

            # 3. 解析反思结果
            parsed = self._parse_reflection(response.content)

            reflection.is_successful = parsed.get('is_successful', trace.execution_success)
            reflection.failure_type = FailureType(parsed.get('failure_type', 'none'))
            reflection.key_insights = parsed.get('key_insights', [])
            reflection.error_patterns = parsed.get('error_patterns', [])

            # 4. 评估策略有效性
            for eval_data in parsed.get('strategy_evaluations', []):
                evaluation = StrategyEvaluation(
                    entry_id=eval_data.get('entry_id', ''),
                    is_helpful=eval_data.get('is_helpful', True),
                    reason=eval_data.get('reason', '')
                )
                reflection.strategy_evaluations.append(evaluation)

            # 如果 LLM 没有评估策略，使用简单规则
            if not reflection.strategy_evaluations and trace.retrieved_entries:
                reflection.strategy_evaluations = self._simple_strategy_evaluation(trace)

            cprint(f"✅ 反思完成: 洞察 {len(reflection.key_insights)} 个, 错误模式 {len(reflection.error_patterns)} 个", "green")

        except Exception as e:
            cprint(f"❌ Reflector 失败: {e}", "red")
            # 降级：简单的规则判断
            reflection.is_successful = trace.execution_success
            if not trace.execution_success:
                reflection.failure_type = FailureType.EXECUTION
                reflection.error_patterns = [f"执行失败: {trace.execution_error}"]

            # 简单评估策略
            reflection.strategy_evaluations = self._simple_strategy_evaluation(trace)

        return reflection

    def _parse_reflection(self, llm_output: str) -> dict:
        """解析反思 JSON"""
        try:
            start = llm_output.find('{')
            end = llm_output.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = llm_output[start:end]
                return json.loads(json_str)

        except Exception as e:
            cprint(f"⚠️  解析反思失败: {e}", "yellow")

        return {}

    def _simple_strategy_evaluation(self, trace: ExecutionTrace) -> list:
        """
        简单的策略评估（降级方案）

        规则：
        - 如果执行成功且盈利 → 所有策略标记为 helpful
        - 如果执行失败或亏损 → 所有策略标记为 harmful
        """
        evaluations = []

        if not trace.retrieved_entries:
            return evaluations

        # 判断是否成功
        is_successful = trace.execution_success

        # 如果有 PnL 信息，使用盈亏判断
        if trace.pnl is not None:
            is_successful = is_successful and (float(trace.pnl) >= 0)

        # 创建评估
        for entry_id in trace.retrieved_entries[:5]:  # 只评估前 5 个
            evaluations.append(StrategyEvaluation(
                entry_id=entry_id,
                is_helpful=is_successful,
                reason="基于执行结果的简单评估"
            ))

        return evaluations
