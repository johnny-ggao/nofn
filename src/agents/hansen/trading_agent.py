"""
Hansen Trading Agent - Tool-based 实现
"""
import asyncio
from termcolor import cprint
from typing import Dict, Optional, List, Any
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from ...adapters.base import BaseExchangeAdapter
from ...tools import (
    ALL_TOOLS,
    set_adapter as set_tools_adapter
)


class TradingAgent:
    """
    Hansen 策略交易智能体 - Tool-based 实现
    """

    def __init__(
        self,
        adapter: BaseExchangeAdapter,
        llm: Optional[ChatOpenAI] = None,
        config: Optional[Dict] = None,
    ):
        self.adapter = adapter
        self.llm = llm
        self.config = config or {}
        self.symbols = self.config.get("symbols", ["BTC/USDC:USDC"])

        # 设置工具适配器
        set_tools_adapter(adapter)
        self.tools = ALL_TOOLS

        # 加载 Hansen 提示词
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "hansen1.txt"
        try:
            system_prompt = prompt_path.read_text(encoding="utf-8")
        except Exception as e:
            cprint(f"   ⚠️ Failed to load prompt: {e}", "yellow")
            return

        self.agent_graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=False,
        )

        cprint(f"✅ TradingAgent initialized", "green")
        cprint(f"   Symbols: {self.symbols}", "cyan")
        cprint(f"   Tools: {len(self.tools)} available", "cyan")

    # ==================== 核心方法 ====================

    async def analyze_and_decide(self, iteration: int, custom_instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        让 Agent 分析市场并做出决策

        注意：system_prompt (hansen1.txt) 已包含完整策略，这里只需简单触发

        Args:
            custom_instruction: 自定义指令（可选）
            iteration:

        Returns:
            Dict: Agent 的决策结果
        """
        try:
            # 简洁的用户消息（system_prompt 已包含完整策略）
            if custom_instruction is None:
                symbols_str = ", ".join(self.symbols)
                user_message = f"请分析 {symbols_str} 的市场情况并根据策略执行交易。"
            else:
                user_message = custom_instruction

            cprint("=" * 70, "cyan")
            cprint(f"🤖 Agent 开始分析( Iteration {iteration + 1})...", "cyan")
            cprint("=" * 70, "cyan")

            # 调用 Agent
            result = await self.agent_graph.ainvoke({
                "messages": [HumanMessage(content=user_message)]
            })

            # 提取响应
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                output_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
            else:
                output_text = "No response from agent"

            cprint(f"\n{output_text}\n", "white")
            cprint("=" * 70, "green")
            cprint("✅ Agent 分析完成", "green")
            cprint("=" * 70, "green")

            return {
                "success": True,
                "output": output_text,
                "messages": messages,
            }

        except Exception as e:
            cprint(f"❌ Agent execution error: {e}", "red")
            return {
                "success": False,
                "error": str(e)
            }

    async def run_loop(
        self,
        interval_seconds: int = 180,
        max_iterations: Optional[int] = None,
    ):
        """
        持续运行交易循环

        Args:
            interval_seconds: 循环间隔(秒)
            max_iterations: 最大迭代次数 (None = 无限)
        """
        iteration = 0

        try:
            # 初始化交易所连接
            await self.adapter.initialize()
            cprint("🚀 TradingAgent started", "green")
            cprint(f"   Workflow: Tool-calling agent (autonomous)", "cyan")
            cprint(f"   Tools: {len(self.tools)} available", "cyan")

            while True:
                if max_iterations and iteration >= max_iterations:
                    cprint(f"✓ Reached max iterations ({max_iterations})", "green")
                    break

                cprint("\n" + "=" * 70, "cyan")
                cprint(f"📊 Iteration {iteration + 1}", "cyan")
                cprint("=" * 70 + "\n", "cyan")

                # Agent 自主分析和决策
                result = await self.analyze_and_decide(iteration=iteration)

                if not result.get("success"):
                    cprint(f"⚠️ Iteration {iteration + 1} failed: {result.get('error')}", "yellow")

                cprint(f"\n✓ Iteration {iteration + 1} complete\n", "green")

                iteration += 1

                if max_iterations is None or iteration < max_iterations:
                    cprint(f"⏳ Waiting {interval_seconds}s until next iteration...\n", "cyan")
                    await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            cprint("⚠️ Received interrupt signal, stopping...", "yellow")

        except Exception as e:
            cprint(f"❌ Loop error: {e}", "red")
            raise

        finally:
            await self.adapter.close()
            cprint("🏁 ToolBasedAgent stopped", "green")