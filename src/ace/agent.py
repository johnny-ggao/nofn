"""
ACE Agent - 主控制器

基于 LangGraph 编排 Generator-Reflector-Curator 循环，实现自进化交易智能体
"""

import asyncio
from typing import List, Optional

from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from termcolor import cprint

from .storage import ContextStore
from .utils import EmbeddingService
from .core import Generator, Reflector, Curator
from .graph.state import ACEState
from .graph.nodes import (
    create_generator_node,
    create_reflector_node,
    create_curator_node,
    create_summary_node,
    create_maintenance_node
)


class ACEAgent:
    """
    ACE Trading Agent

    基于 Agentic Context Engineering 框架 + LangGraph 的自进化交易智能体
    """

    def __init__(
        self,
        llm,
        exchange_adapter,
        api_key: str,
        db_path: str = "data/ace.db",
        base_url: Optional[str] = None,
        embedding_provider: str = "openai",
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        checkpoint_path: str = "data/ace_checkpoints.db"
    ):
        """
        初始化 ACE Agent

        Args:
            llm: LangChain LLM 实例
            exchange_adapter: 交易所适配器（HyperliquidAdapter）
            api_key: LLM API Key
            db_path: 数据库路径
            base_url: LLM API 端点（可选）
            embedding_provider: Embedding 提供商 (openai/zhipu/ollama/dashscope)
            embedding_api_key: Embedding API Key（默认使用 api_key）
            embedding_base_url: Embedding API URL（可选）
            embedding_model: Embedding 模型（可选）
            checkpoint_path: LangGraph 检查点数据库路径
        """
        # 初始化存储层
        self.context_store = ContextStore(db_path)

        # 初始化向量化服务
        self.embedding_service = EmbeddingService(
            api_key=embedding_api_key or api_key,
            base_url=embedding_base_url,
            provider=embedding_provider,
            model=embedding_model
        )

        # 初始化三大核心模块
        self.generator = Generator(
            llm=llm,
            exchange_adapter=exchange_adapter,
            context_store=self.context_store,
            embedding_service=self.embedding_service
        )

        self.reflector = Reflector(llm=llm)

        self.curator = Curator(
            context_store=self.context_store,
            embedding_service=self.embedding_service
        )

        # 检查点路径
        self.checkpoint_path = checkpoint_path

        # LangGraph 工作流（延迟构建）
        self.workflow = None
        self.app = None
        self.checkpointer = None  # 用于保存 AsyncSqliteSaver 实例
        self.checkpointer_cm = None  # 上下文管理器

        cprint("==" * 35, "green")
        cprint("✅ ACE Agent (LangGraph) 初始化完成", "green", attrs=['bold'])
        cprint("==" * 35, "green")

    def _build_workflow(self):
        """构建 LangGraph 工作流"""
        if self.workflow is not None:
            return  # 已经构建过了

        # 创建节点函数
        generator_node = create_generator_node(self.generator, self.context_store)
        reflector_node = create_reflector_node(self.reflector, self.context_store)
        curator_node = create_curator_node(self.curator)
        summary_node = create_summary_node()
        maintenance_node = create_maintenance_node(self.context_store)

        # 创建状态图
        workflow = StateGraph(ACEState)

        # 添加节点
        workflow.add_node("generator", generator_node)
        workflow.add_node("reflector", reflector_node)
        workflow.add_node("curator", curator_node)
        workflow.add_node("summary", summary_node)
        workflow.add_node("maintenance", maintenance_node)

        # 设置入口点
        workflow.set_entry_point("generator")

        # 添加边（定义执行顺序）
        workflow.add_edge("generator", "reflector")
        workflow.add_edge("reflector", "curator")
        workflow.add_edge("curator", "summary")
        workflow.add_edge("summary", "maintenance")
        workflow.add_edge("maintenance", END)

        self.workflow = workflow

    async def _compile_app(self):
        """编译 LangGraph 应用（带检查点支持）"""
        if self.app is not None:
            return  # 已经编译过了

        # 构建工作流
        self._build_workflow()

        checkpoint_path = Path(self.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.checkpointer_cm = AsyncSqliteSaver.from_conn_string(str(checkpoint_path))
        self.checkpointer = await self.checkpointer_cm.__aenter__()  # 获取实际的 saver 对象

        # 编译应用
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

        cprint(f"✅ LangGraph 工作流已编译（检查点: {checkpoint_path}）", "green")

    async def run_loop(
        self,
        symbols: List[str],
        interval_seconds: int = 180,
        max_iterations: Optional[int] = None
    ):
        """
        运行主循环

        Args:
            symbols: 交易对列表
            interval_seconds: 迭代间隔（秒）
            max_iterations: 最大迭代次数（None 表示无限循环）
        """
        try:
            # 编译应用
            await self._compile_app()

            iteration = 0

            cprint("\n" + "==" * 35, "cyan")
            cprint("🚀 ACE Trading Agent 启动 (LangGraph)", "cyan", attrs=['bold'])
            cprint("==" * 35, "cyan")

            # 显示知识库统计
            stats = self.context_store.get_statistics()
            cprint(f"\n📚 知识库统计:", "cyan")
            cprint(f"  - 总条目: {stats['total_entries']} 个", "white")
            cprint(f"  - 执行轨迹: {stats['total_traces']} 次", "white")
            cprint(f"  - 反思记录: {stats['total_reflections']} 次", "white")

            if stats['by_type']:
                cprint(f"\n  条目类型分布:", "white")
                for entry_type, count in stats['by_type'].items():
                    cprint(f"    - {entry_type}: {count} 个", "white")

            cprint(f"\n⚙️  运行参数:", "cyan")
            cprint(f"  - 交易对: {', '.join(symbols)}", "white")
            cprint(f"  - 迭代间隔: {interval_seconds} 秒", "white")
            cprint(f"  - 最大迭代: {max_iterations if max_iterations else '无限'} 次", "white")

            while max_iterations is None or iteration < max_iterations:
                iteration += 1

                cprint("\n" + "==" * 35, "yellow")
                cprint(f"🔄 迭代 #{iteration}", "yellow", attrs=['bold'])
                cprint("==" * 35, "yellow")

                try:
                    # 构建初始状态
                    initial_state: ACEState = {
                        "symbols": symbols,
                        "iteration": iteration,
                        "trace": None,
                        "reflection": None,
                        "updated_entries": [],
                        "should_continue": True,
                        "max_iterations": max_iterations,
                        "errors": []
                    }

                    # 执行工作流
                    config = {
                        "configurable": {
                            "thread_id": f"ace-{iteration}"
                        }
                    }

                    # 流式执行（可以实时看到每个节点的输出）
                    async for event in self.app.astream(initial_state, config):
                        # event 是 {节点名: 节点输出} 的字典
                        # LangGraph 会自动合并状态
                        pass

                    # 等待下次迭代
                    if max_iterations is None or iteration < max_iterations:
                        cprint(f"\n⏸️  等待 {interval_seconds} 秒后继续下一次迭代...", "white")
                        await asyncio.sleep(interval_seconds)

                except KeyboardInterrupt:
                    cprint("\n" + "==" * 35, "yellow")
                    cprint("⚠️  收到中断信号，正在停止...", "yellow", attrs=['bold'])
                    cprint("==" * 35, "yellow")
                    break

                except Exception as e:
                    cprint(f"\n❌ 迭代 #{iteration} 失败: {e}", "red", attrs=['bold'])
                    import traceback
                    traceback.print_exc()
                    # 继续下一次迭代
                    cprint(f"\n⏭️  {interval_seconds} 秒后继续下一次迭代...", "yellow")
                    await asyncio.sleep(interval_seconds)

            # 结束统计
            cprint("\n" + "==" * 35, "cyan")
            cprint("👋 ACE Agent 停止", "cyan", attrs=['bold'])
            cprint("==" * 35, "cyan")

            # 最终统计
            final_stats = self.context_store.get_statistics()
            cprint(f"\n📊 最终统计:", "green")
            cprint(f"  - 知识库条目: {final_stats['total_entries']} 个", "white")
            cprint(f"  - 执行轨迹: {final_stats['total_traces']} 次", "white")
            cprint(f"  - 反思记录: {final_stats['total_reflections']} 次", "white")

            cprint(f"\n💾 数据已保存到:", "green")
            cprint(f"  - 知识库: {self.context_store.db_path}", "white")
            cprint(f"  - 检查点: {self.checkpoint_path}", "white")

        finally:
            # 关闭 checkpointer
            if self.checkpointer_cm is not None:
                try:
                    await self.checkpointer_cm.__aexit__(None, None, None)
                    self.checkpointer = None
                    self.checkpointer_cm = None
                except Exception as e:
                    cprint(f"⚠️  关闭检查点时出错: {e}", "yellow")

    async def single_iteration(self, symbols: List[str]) -> dict:
        """
        执行单次迭代（用于测试）

        Returns:
            包含 trace, reflection, updated_entries 的字典
        """
        try:
            # 编译应用
            await self._compile_app()

            # 构建初始状态
            initial_state: ACEState = {
                "symbols": symbols,
                "iteration": 1,
                "trace": None,
                "reflection": None,
                "updated_entries": [],
                "should_continue": False,  # 单次执行
                "max_iterations": 1,
                "errors": []
            }

            # 执行工作流
            config = {"configurable": {"thread_id": "ace-single"}}

            final_state = None
            async for event in self.app.astream(initial_state, config):
                # 保存最后的状态
                for node_name, node_output in event.items():
                    if node_output:
                        final_state = node_output

            return {
                'trace': final_state.get('trace') if final_state else None,
                'reflection': final_state.get('reflection') if final_state else None,
                'updated_entries': final_state.get('updated_entries', []) if final_state else []
            }

        finally:
            # 关闭 checkpointer
            if self.checkpointer_cm is not None:
                try:
                    await self.checkpointer_cm.__aexit__(None, None, None)
                    self.checkpointer = None
                    self.checkpointer_cm = None
                except Exception as e:
                    cprint(f"⚠️  关闭检查点时出错: {e}", "yellow")
