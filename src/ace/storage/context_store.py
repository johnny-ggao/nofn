"""
Context Store - ACE 知识库的存储和检索层

负责：
1. 持久化 ContextEntry
2. 检索相关策略（基于相似度）
3. 去重和合并
4. 清理过期条目
"""

import json
import pickle
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from termcolor import cprint

from ..models import ContextEntry, EntryType, ExecutionTrace, Reflection


class ContextStore:
    """
    ACE 知识库

    使用 SQLite 存储，支持：
    - ContextEntry 的 CRUD
    - 基于 embedding 的相似度检索
    - 增量式更新
    """

    def __init__(self, db_path: str = "data/ace.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """初始化数据库结构"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Playbook Entries 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS playbook_entries (
                entry_id TEXT PRIMARY KEY,
                entry_type TEXT NOT NULL,
                content TEXT NOT NULL,
                helpful_count INTEGER DEFAULT 0,
                harmful_count INTEGER DEFAULT 0,
                neutral_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                last_used TEXT,
                last_updated TEXT NOT NULL,
                source_trace_ids TEXT,  -- JSON array
                tags TEXT,              -- JSON array
                embedding BLOB          -- Numpy array (pickled)
            )
        ''')

        # Execution Traces 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS execution_traces (
                trace_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                market_data TEXT,        -- JSON
                account_state TEXT,      -- JSON
                retrieved_entries TEXT,  -- JSON array
                decisions TEXT,          -- JSON array (多个决策)
                execution_success INTEGER,
                execution_results TEXT,  -- JSON array (多个执行结果)
                execution_errors TEXT,   -- JSON array (多个错误)
                account_change TEXT,     -- JSON
                raw_llm_output TEXT
            )
        ''')

        # Reflections 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reflections (
                reflection_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                is_successful INTEGER,
                failure_type TEXT,
                strategy_evaluations TEXT,  -- JSON array
                key_insights TEXT,          -- JSON array
                error_patterns TEXT,        -- JSON array
                improvement_suggestions TEXT,  -- JSON array
                reflection_text TEXT,
                market_conditions TEXT,     -- JSON
                FOREIGN KEY (trace_id) REFERENCES execution_traces(trace_id)
            )
        ''')

        # 索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entry_type ON playbook_entries(entry_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_confidence ON playbook_entries(helpful_count, harmful_count)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_used ON playbook_entries(last_used)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trace_timestamp ON execution_traces(timestamp)')

        # 启用 WAL 模式
        cursor.execute('PRAGMA journal_mode=WAL')

        conn.commit()
        conn.close()

        cprint(f"✅ Context Store 初始化完成: {self.db_path}", "green")

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        return sqlite3.connect(str(self.db_path), timeout=30.0)

    # ========== ContextEntry 操作 ==========

    def add_entry(self, entry: ContextEntry) -> bool:
        """添加新条目"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # 序列化 embedding
            embedding_blob = pickle.dumps(entry.embedding) if entry.embedding else None

            cursor.execute('''
                INSERT INTO playbook_entries (
                    entry_id, entry_type, content,
                    helpful_count, harmful_count, neutral_count,
                    created_at, last_used, last_updated,
                    source_trace_ids, tags, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.entry_id,
                entry.entry_type.value,
                entry.content,
                entry.helpful_count,
                entry.harmful_count,
                entry.neutral_count,
                entry.created_at.isoformat(),
                entry.last_used.isoformat() if entry.last_used else None,
                entry.last_updated.isoformat(),
                json.dumps(entry.source_trace_ids),
                json.dumps(entry.tags),
                embedding_blob,
            ))

            conn.commit()
            return True

        except sqlite3.IntegrityError:
            # 条目已存在
            return False
        finally:
            conn.close()

    def get_entry(self, entry_id: str) -> Optional[ContextEntry]:
        """获取指定条目"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM playbook_entries WHERE entry_id = ?', (entry_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_entry(row)

    def update_entry(self, entry: ContextEntry) -> bool:
        """更新条目"""
        conn = self._get_connection()
        cursor = conn.cursor()

        embedding_blob = pickle.dumps(entry.embedding) if entry.embedding else None

        cursor.execute('''
            UPDATE playbook_entries SET
                entry_type = ?,
                content = ?,
                helpful_count = ?,
                harmful_count = ?,
                neutral_count = ?,
                last_used = ?,
                last_updated = ?,
                source_trace_ids = ?,
                tags = ?,
                embedding = ?
            WHERE entry_id = ?
        ''', (
            entry.entry_type.value,
            entry.content,
            entry.helpful_count,
            entry.harmful_count,
            entry.neutral_count,
            entry.last_used.isoformat() if entry.last_used else None,
            datetime.now().isoformat(),
            json.dumps(entry.source_trace_ids),
            json.dumps(entry.tags),
            embedding_blob,
            entry.entry_id,
        ))

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    def delete_entry(self, entry_id: str) -> bool:
        """删除条目"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM playbook_entries WHERE entry_id = ?', (entry_id,))
        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    def get_all_entries(self, entry_type: Optional[EntryType] = None) -> List[ContextEntry]:
        """获取所有条目（可按类型筛选）"""
        conn = self._get_connection()
        cursor = conn.cursor()

        if entry_type:
            cursor.execute(
                'SELECT * FROM playbook_entries WHERE entry_type = ? ORDER BY last_updated DESC',
                (entry_type.value,)
            )
        else:
            cursor.execute('SELECT * FROM playbook_entries ORDER BY last_updated DESC')

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_entry(row) for row in rows]

    def _row_to_entry(self, row: tuple) -> ContextEntry:
        """将数据库行转换为 ContextEntry"""
        embedding = pickle.loads(row[11]) if row[11] else None

        return ContextEntry(
            entry_id=row[0],
            entry_type=EntryType(row[1]),
            content=row[2],
            helpful_count=row[3],
            harmful_count=row[4],
            neutral_count=row[5],
            created_at=datetime.fromisoformat(row[6]),
            last_used=datetime.fromisoformat(row[7]) if row[7] else None,
            last_updated=datetime.fromisoformat(row[8]),
            source_trace_ids=json.loads(row[9]) if row[9] else [],
            tags=json.loads(row[10]) if row[10] else [],
            embedding=embedding,
        )

    # ========== 相似度检索 ==========

    def retrieve_similar_entries(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_confidence: float = 0.3,
        entry_type: Optional[EntryType] = None
    ) -> List[Tuple[ContextEntry, float]]:
        """
        检索相似的条目

        Args:
            query_embedding: 查询向量
            top_k: 返回前 K 个
            min_confidence: 最低置信度阈值
            entry_type: 筛选条目类型

        Returns:
            [(entry, similarity_score), ...] 按相似度降序
        """
        entries = self.get_all_entries(entry_type)

        # 过滤低置信度
        entries = [e for e in entries if e.confidence >= min_confidence and e.embedding is not None]

        if not entries:
            return []

        # 计算余弦相似度
        query_vec = np.array(query_embedding)
        results = []

        for entry in entries:
            entry_vec = np.array(entry.embedding)
            similarity = self._cosine_similarity(query_vec, entry_vec)
            results.append((entry, similarity))

        # 排序并返回 top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ========== ExecutionTrace 操作 ==========

    def save_trace(self, trace: ExecutionTrace):
        """保存执行轨迹"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO execution_traces (
                trace_id, timestamp, market_data, account_state,
                retrieved_entries, decisions,
                execution_success, execution_results, execution_errors,
                account_change, raw_llm_output
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trace.trace_id,
            trace.timestamp.isoformat(),
            json.dumps(trace.market_data),
            json.dumps(trace.account_state),
            json.dumps(trace.retrieved_entries),
            json.dumps([d.to_dict() for d in trace.decisions]) if trace.decisions else None,
            1 if trace.execution_success else 0,
            json.dumps(trace.execution_results),
            json.dumps(trace.execution_errors),
            json.dumps(trace.account_change),
            trace.raw_llm_output,
        ))

        conn.commit()
        conn.close()

    # ========== Reflection 操作 ==========

    def save_reflection(self, reflection: Reflection):
        """保存反思结果"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO reflections (
                reflection_id, timestamp, trace_id,
                is_successful, failure_type,
                strategy_evaluations, key_insights, error_patterns,
                improvement_suggestions, reflection_text, market_conditions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            reflection.reflection_id,
            reflection.timestamp.isoformat(),
            reflection.trace_id,
            1 if reflection.is_successful else 0,
            reflection.failure_type.value,
            json.dumps([e.to_dict() for e in reflection.strategy_evaluations]),
            json.dumps(reflection.key_insights),
            json.dumps(reflection.error_patterns),
            json.dumps(reflection.improvement_suggestions),
            reflection.reflection_text,
            json.dumps(reflection.market_conditions),
        ))

        conn.commit()
        conn.close()

    # ========== 维护操作 ==========

    def prune_low_confidence_entries(self, threshold: float = 0.2) -> int:
        """删除低置信度条目"""
        entries = self.get_all_entries()
        deleted_count = 0

        for entry in entries:
            if entry.confidence < threshold:
                self.delete_entry(entry.entry_id)
                deleted_count += 1

        return deleted_count

    def archive_old_entries(self, days: int = 90) -> int:
        """归档长期未使用的条目"""
        cutoff_date = datetime.now() - timedelta(days=days)
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            DELETE FROM playbook_entries
            WHERE last_used < ? OR (last_used IS NULL AND created_at < ?)
        ''', (cutoff_date.isoformat(), cutoff_date.isoformat()))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count

    def get_statistics(self) -> dict:
        """获取知识库统计信息"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM playbook_entries')
        total_entries = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM execution_traces')
        total_traces = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM reflections')
        total_reflections = cursor.fetchone()[0]

        # 按类型统计
        cursor.execute('SELECT entry_type, COUNT(*) FROM playbook_entries GROUP BY entry_type')
        by_type = dict(cursor.fetchall())

        conn.close()

        return {
            'total_entries': total_entries,
            'total_traces': total_traces,
            'total_reflections': total_reflections,
            'by_type': by_type,
        }

    def get_all_entries(self, limit: Optional[int] = None) -> List[ContextEntry]:
        """获取所有知识条目"""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = '''
            SELECT entry_id, entry_type, content, helpful_count, harmful_count, neutral_count,
                   created_at, last_used, last_updated, source_trace_ids, tags, embedding
            FROM playbook_entries
            ORDER BY last_updated DESC
        '''

        if limit:
            query += f' LIMIT {limit}'

        cursor.execute(query)
        rows = cursor.fetchall()

        entries = []
        for row in rows:
            entry = self._row_to_entry(row)
            entries.append(entry)

        conn.close()
        return entries

    def delete_entry(self, entry_id: str) -> bool:
        """删除知识条目"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM playbook_entries WHERE entry_id = ?', (entry_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

    def get_recent_traces(self, limit: int = 10) -> List[ExecutionTrace]:
        """获取最近的执行轨迹"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT trace_id, timestamp, market_data, account_state, retrieved_entries,
                   decisions, execution_success, execution_results, execution_errors, account_change, raw_llm_output
            FROM execution_traces
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        traces = []

        for row in rows:
            trace = self._row_to_trace(row)
            traces.append(trace)

        conn.close()
        return traces

    def get_recent_reflections(self, limit: int = 10) -> List[Reflection]:
        """获取最近的反思记录"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT reflection_id, timestamp, trace_id, is_successful, failure_type,
                   strategy_evaluations, key_insights, error_patterns, improvement_suggestions,
                   reflection_text, market_conditions
            FROM reflections
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        reflections = []

        for row in rows:
            reflection = self._row_to_reflection(row)
            reflections.append(reflection)

        conn.close()
        return reflections
