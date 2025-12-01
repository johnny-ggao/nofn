import json
import re
from typing import Dict, Any, Optional, List
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from termcolor import cprint

from ..engine.market_snapshot import MarketSnapshot


class TradingAgent:
    """
    交易 Agent

    负责：
    1. 分析市场数据并做出交易决策
    2. 自我评估决策质量
    3. 学习和调整策略
    """

    def __init__(
        self,
        model_provider: str = "openai",
        model_id: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        system_prompt_path: Optional[str] = None,
    ):
        """
        初始化 Trading Agent

        Args:
            model_provider: 模型提供商 (openai, anthropic)
            model_id: 模型 ID
            api_key: API 密钥
            base_url: API 基础 URL (可选，用于自定义端点)
            temperature: 温度参数
            system_prompt_path: 系统提示词文件路径
        """
        self.model_provider = model_provider
        self.model_id = model_id
        self.temperature = temperature

        self.llm = ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

        self.system_prompt = self._load_system_prompt(system_prompt_path)

        cprint(f"✅ TradingAgent 初始化完成 ({model_provider}/{model_id})", "green")

    @staticmethod
    def _load_system_prompt(prompt_path: Optional[str]) -> str:
        """加载系统提示词"""
        if prompt_path and Path(prompt_path).exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError("❌ 加载系统提示词失败")

    async def make_decision(
        self,
        market_snapshot: MarketSnapshot,
        memory_context: Optional[str] = None,
        recent_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        做出交易决策

        Args:
            market_snapshot: 市场快照
            memory_context: 历史记忆上下文
            recent_trades: 最近交易记录列表

        Returns:
            决策结果字典
        """
        # 构建提示词
        user_prompt = self._build_decision_prompt(market_snapshot, memory_context, recent_trades)

        # 调用 LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw_response = response.content

            cprint("📝 LLM 决策响应已收到", "cyan")

            # 解析 JSON 响应
            decision = self._parse_decision(raw_response)
            decision['raw_response'] = raw_response

            return decision

        except Exception as e:
            cprint(f"❌ LLM 调用失败: {e}", "red")
            return {
                'decision_type': 'wait',
                'signals': [],
                'analysis': f'LLM 调用失败: {str(e)}',
                'error': str(e),
            }

    def _build_decision_prompt(
        self,
        market_snapshot: MarketSnapshot,
        memory_context: Optional[str],
        recent_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """构建决策提示词"""
        lines = []

        # 使用 MarketSnapshot 的 to_text() 方法获取完整市场数据
        # 包含多时间框架指标 (4H/1H/15M)、永续合约指标、持仓信息等
        lines.append(market_snapshot.to_text())
        lines.append("")

        # 最近交易记录
        if recent_trades:
            lines.append("## 最近交易记录")
            lines.append("")

            # 交易类型映射
            trade_type_map = {
                'open': '开仓',
                'close': '平仓',
                'add': '加仓',
                'reduce': '减仓',
            }

            # 统计盈亏
            total_pnl = 0.0
            win_count = 0
            loss_count = 0

            for trade in recent_trades:
                trade_type = trade.get('trade_type', 'N/A')
                trade_type_cn = trade_type_map.get(trade_type, trade_type)
                closed_pnl = trade.get('closed_pnl')

                line = f"- {trade.get('symbol', 'N/A')} | {trade_type_cn} | {trade.get('side', 'N/A').upper()} | 价格: ${trade.get('price', 0):.2f} | 数量: {trade.get('amount', 0)}"

                if trade_type == 'close' and closed_pnl is not None:
                    pnl_sign = "+" if closed_pnl >= 0 else ""
                    line += f" | **盈亏: {pnl_sign}${closed_pnl:.2f}**"
                    total_pnl += closed_pnl
                    if closed_pnl >= 0:
                        win_count += 1
                    else:
                        loss_count += 1

                lines.append(line)

            # 添加统计摘要
            close_count = win_count + loss_count
            if close_count > 0:
                lines.append("")
                win_rate = win_count / close_count * 100
                pnl_sign = "+" if total_pnl >= 0 else ""
                lines.append(f"**最近 {close_count} 笔平仓统计**: 胜率 {win_rate:.0f}% ({win_count}胜/{loss_count}负), 总盈亏 {pnl_sign}${total_pnl:.2f}")

            lines.append("")

        # 历史经验
        if memory_context:
            lines.append("## 历史经验")
            lines.append(memory_context)
            lines.append("")

        # 输出要求
        lines.append("## 输出要求")
        lines.append("")
        lines.append("请分析以上市场数据并做出交易决策。输出 JSON 格式：")
        lines.append("")
        lines.append("```json")
        lines.append("{")
        lines.append('  "analysis": "你的分析（2-3句话）",')
        lines.append('  "decision_type": "trade | hold | wait",')
        lines.append('  "signals": [')
        lines.append('    {')
        lines.append('      "action": "open_long | open_short | close_long | close_short | hold",')
        lines.append('      "symbol": "BTC/USDC:USDC",')
        lines.append('      "amount": 0.001,')
        lines.append('      "leverage": 3,')
        lines.append('      "stop_loss": 88000.0,')
        lines.append('      "take_profit": 96000.0,')
        lines.append('      "confidence": 85,')
        lines.append('      "reason": "具体理由"')
        lines.append('    }')
        lines.append('  ]')
        lines.append('}')
        lines.append("```")

        return "\n".join(lines)

    def _parse_decision(self, text: str) -> Dict[str, Any]:
        """解析决策响应"""
        result = {
            'analysis': text,
            'decision_type': 'wait',
            'signals': [],
        }

        try:
            # 提取 JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                result.update({
                    'analysis': data.get('analysis', result['analysis']),
                    'decision_type': data.get('decision_type', 'wait'),
                    'signals': data.get('signals', []),
                })
        except (json.JSONDecodeError, AttributeError) as e:
            cprint(f"⚠️  解析决策 JSON 失败: {e}", "yellow")

        return result

    async def evaluate_and_learn(
        self,
        decision: Dict[str, Any],
        execution_results: list,
        account_info: Optional[Dict],
        market_snapshot: MarketSnapshot,
    ) -> Dict[str, Any]:
        """
        评估决策并学习

        Args:
            decision: 原始决策
            execution_results: 执行结果
            account_info: 账户信息
            market_snapshot: 市场快照

        Returns:
            评估结果字典
        """
        # 构建评估提示词
        prompt = self._build_evaluation_prompt(
            decision, execution_results, account_info, market_snapshot
        )

        # 调用 LLM
        messages = [
            SystemMessage(content="你是一个交易决策评估专家。客观评估交易决策的质量并提供改进建议。"),
            HumanMessage(content=prompt)
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw_response = response.content

            cprint("📊 LLM 评估响应已收到", "cyan")

            # 解析评估结果
            evaluation = self._parse_evaluation(raw_response)
            evaluation['raw_response'] = raw_response

            return evaluation

        except Exception as e:
            cprint(f"❌ 评估失败: {e}", "red")
            return {
                'analysis': f'评估失败: {str(e)}',
                'quality_score': 50,
                'lessons': [],
                'error': str(e),
            }

    @staticmethod
    def _build_evaluation_prompt(
        decision: Dict[str, Any],
        execution_results: list,
        account_info: Optional[Dict],
        market_snapshot: MarketSnapshot,
    ) -> str:
        """构建评估提示词"""
        lines = ["# 决策评估", ""]

        # 原始决策
        lines.append("## 原始决策")
        lines.append(f"决策类型: {decision.get('decision_type')}")
        lines.append(f"分析: {decision.get('analysis', 'N/A')}")
        lines.append("")

        # 执行结果
        lines.append("## 执行结果")
        for i, result in enumerate(execution_results, 1):
            signal = result.get('signal', {})
            res = result.get('result', {})
            lines.append(f"{i}. {signal.get('action')} {signal.get('symbol')}: {'成功' if res.get('success') else '失败'}")

        lines.append("")

        # 账户状态
        if account_info:
            lines.append("## 账户状态")
            balance = account_info.get('balance', {})
            lines.append(f"总余额: ${balance.get('total', 0):.2f}")
            lines.append(f"可用余额: ${balance.get('available', 0):.2f}")

            stats = account_info.get('statistics', {})
            if stats.get('total_positions', 0) > 0:
                lines.append(f"总交易: {stats.get('total_positions', 0)}笔")
                lines.append(f"胜率: {stats.get('win_rate', 0) * 100:.1f}%")
                lines.append(f"总盈亏: ${stats.get('total_pnl', 0):.2f}")

        lines.append("")

        # 输出要求
        lines.append("## 评估要求")
        lines.append("")
        lines.append("请评估这次决策的质量，输出 JSON 格式：")
        lines.append("")
        lines.append("```json")
        lines.append("{")
        lines.append('  "analysis": "整体评估（2-3句话）",')
        lines.append('  "quality_score": 0-100,')
        lines.append('  "lessons": ["具体经验1", "具体经验2"],')
        lines.append('  "sl_tp_adjustment": {')
        lines.append('    "action": "tighten | loosen | keep",')
        lines.append('    "reason": "调整原因",')
        lines.append('    "stop_loss_multiplier": 1.0,')
        lines.append('    "take_profit_multiplier": 1.0')
        lines.append('  }')
        lines.append('}')
        lines.append("```")

        return "\n".join(lines)

    @staticmethod
    def _parse_evaluation(text: str) -> Dict[str, Any]:
        """解析评估响应"""
        result = {
            'analysis': text,
            'quality_score': 50,
            'lessons': [],
            'sl_tp_adjustment': {},
        }

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                result.update({
                    'analysis': data.get('analysis', result['analysis']),
                    'quality_score': data.get('quality_score', 50),
                    'lessons': data.get('lessons', []),
                    'sl_tp_adjustment': data.get('sl_tp_adjustment', {}),
                })

                # 输出止盈止损调整日志
                sl_tp_adj = result.get('sl_tp_adjustment', {})
                if sl_tp_adj and sl_tp_adj.get('action') != 'keep':
                    action_text = {'tighten': '收紧', 'loosen': '放宽', 'keep': '保持'}.get(sl_tp_adj.get('action'), '未知')
                    cprint(f"📊 止盈止损调整建议: {action_text}", "cyan")
                    cprint(f"   原因: {sl_tp_adj.get('reason', '无')}", "cyan")

        except (json.JSONDecodeError, AttributeError) as e:
            cprint(f"⚠️  解析评估 JSON 失败: {e}", "yellow")

        return result