"""
Curator - ACE 策展者模块

职责：
1. 更新现有条目的元数据
2. 创建新条目（来自洞察）
3. 去重和合并
4. 维护知识库
"""

from typing import List
from termcolor import cprint

from ..models import Reflection, ContextEntry, EntryType
from ..storage import ContextStore
from ..utils import EmbeddingService


class Curator:
    """ACE 策展者"""

    def __init__(
        self,
        context_store: ContextStore,
        embedding_service: EmbeddingService
    ):
        self.context_store = context_store
        self.embedding_service = embedding_service

    async def curate(self, reflection: Reflection) -> List[ContextEntry]:
        """策展知识条目"""
        updated_entries = []

        try:
            # 1. 更新现有条目的元数据
            cprint("📝 更新现有条目的元数据...", "blue")
            for evaluation in reflection.strategy_evaluations:
                entry = self.context_store.get_entry(evaluation.entry_id)
                if entry:
                    if evaluation.is_helpful:
                        entry.mark_helpful()
                        cprint(f"  ✅ {entry.entry_id[:8]}... 标记为有用", "green")
                    else:
                        entry.mark_harmful()
                        cprint(f"  ❌ {entry.entry_id[:8]}... 标记为有害", "red")

                    self.context_store.update_entry(entry)
                    updated_entries.append(entry)

            # 2. 创建新的策略条目（来自 key_insights）
            if reflection.key_insights:
                cprint(f"➕ 创建 {len(reflection.key_insights)} 个新策略条目...", "cyan")
                for insight in reflection.key_insights:
                    new_entry = ContextEntry(
                        entry_type=EntryType.STRATEGY,
                        content=insight,
                        source_trace_ids=[reflection.trace_id]
                    )

                    # 生成 embedding
                    new_entry.embedding = await self.embedding_service.embed(insight)

                    # 去重检查
                    if not await self._is_duplicate(new_entry):
                        self.context_store.add_entry(new_entry)
                        updated_entries.append(new_entry)
                        cprint(f"  ➕ 新增: {insight[:60]}...", "cyan")
                    else:
                        cprint(f"  ⏭️  跳过重复: {insight[:40]}...", "white")

            # 3. 创建错误模式条目
            if reflection.error_patterns:
                cprint(f"⚠️  创建 {len(reflection.error_patterns)} 个错误模式条目...", "yellow")
                for error in reflection.error_patterns:
                    error_entry = ContextEntry(
                        entry_type=EntryType.ERROR_PATTERN,
                        content=error,
                        source_trace_ids=[reflection.trace_id]
                    )

                    error_entry.embedding = await self.embedding_service.embed(error)

                    if not await self._is_duplicate(error_entry):
                        self.context_store.add_entry(error_entry)
                        updated_entries.append(error_entry)
                        cprint(f"  ⚠️  记录: {error[:60]}...", "yellow")
                    else:
                        cprint(f"  ⏭️  跳过重复: {error[:40]}...", "white")

            cprint(f"✅ Curator 完成: 更新/创建 {len(updated_entries)} 个条目", "magenta")

        except Exception as e:
            cprint(f"❌ Curator 失败: {e}", "red")
            import traceback
            traceback.print_exc()

        return updated_entries

    async def _is_duplicate(self, new_entry: ContextEntry) -> bool:
        """
        检查是否重复

        相似度 > 0.95 认为是重复
        """
        if not new_entry.embedding:
            return False

        try:
            similar_entries = self.context_store.retrieve_similar_entries(
                query_embedding=new_entry.embedding,
                top_k=1,
                min_confidence=0.0,
                entry_type=new_entry.entry_type
            )

            if similar_entries:
                top_entry, similarity = similar_entries[0]
                # 相似度 > 0.95 认为是重复
                if similarity > 0.95:
                    cprint(f"    (与 {top_entry.entry_id[:8]}... 相似度 {similarity:.3f})", "white")
                    return True

        except Exception as e:
            cprint(f"⚠️  去重检查失败: {e}", "yellow")

        return False
