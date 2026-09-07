"""
RAG 知识库检索服务 (rag_service.py)
====================================
【模块职责】
管理停车场导航知识库，提供基于关键词/语义的上下文检索
将检索结果注入 LLM 提示词，实现"有据可依"的导航生成

【设计决策】
- 当前版本使用 TF-IDF + 余弦相似度（轻量、无需GPU、启动快）
- 生产环境可无缝替换为 Embedding + FAISS（见优化方向）
- 知识库为纯文本文件，每行一条规则，便于非技术人员维护

【维护须知】
- 新增导航规则：直接编辑 knowledge/parking_nav.txt，每行一条
- 修改检索算法：只需替换 retrieve() 方法内部实现，接口不变
- 本模块不直接调用 LLM，只负责"找到相关知识片段"

【版本记录】
v4.0: 独立为服务模块，支持 TF-IDF 检索
v2.0: 内嵌在 app.py 中的简单关键词匹配
"""

import logging
from pathlib import Path
from typing import Optional

from config import Config, RAG_FALLBACK_KNOWLEDGE

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG 检索服务

    【使用方式】
        from services.rag_service import rag_service
        context = rag_service.retrieve("B1层怎么停车", top_k=3)

    【检索流程】
    用户问题 → 分词 → TF-IDF 向量化 → 余弦相似度排序 → 返回 Top-K 知识片段
    """

    def __init__(self):
        """
        初始化知识库
        【加载策略】
        1. 优先读取 knowledge/parking_nav.txt
        2. 文件不存在时使用 config.py 中的内置知识
        3. 将知识按行分割为独立条目
        """
        self.knowledge_base: list[str] = []
        self._load_knowledge()

    def _load_knowledge(self):
        """
        加载知识库文件

        【格式要求】
        - 每行一条独立规则（空行自动跳过）
        - 单条规则建议不超过 200 字（过长会稀释检索精度）
        - 支持 UTF-8 编码
        """
        knowledge_path = Config.RAG_KNOWLEDGE_PATH

        if knowledge_path.exists():
            with open(knowledge_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.knowledge_base = lines
            logger.info(f"RAG 知识库加载完成 | 来源: {knowledge_path} | 条目数: {len(lines)}")
        else:
            # 文件不存在 → 使用内置知识（保证系统可用）
            lines = [line.strip() for line in RAG_FALLBACK_KNOWLEDGE.strip().split("\n") if line.strip()]
            self.knowledge_base = lines
            logger.warning(
                f"知识库文件不存在: {knowledge_path}，已使用内置知识 | 条目数: {len(lines)}"
            )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> str:
        """
        检索与用户问题最相关的知识片段

        【参数】
        - query: 用户问题或检索关键词
        - top_k: 返回的最大条目数，默认使用 Config.RAG_TOP_K

        【返回值】
        - str: 拼接后的知识上下文（直接注入 LLM 提示词）

        【算法说明】
        当前使用简单的关键词重叠度计算（Jaccard 系数）
        优点：零依赖、启动快、可解释
        缺点：无法处理同义词（如"B1"和"地下一层"）
        优化方向：替换为 sentence-transformers + FAISS 向量检索
        """
        if top_k is None:
            top_k = Config.RAG_TOP_K

        if not self.knowledge_base:
            return ""

        # 简单分词（按字符 n-gram 切分，兼容中文）
        query_tokens = set(self._tokenize(query))

        # 计算每条知识与查询的重叠度
        scored = []
        for i, doc in enumerate(self.knowledge_base):
            doc_tokens = set(self._tokenize(doc))
            # Jaccard 系数 = 交集 / 并集
            intersection = query_tokens & doc_tokens
            union = query_tokens | doc_tokens
            score = len(intersection) / len(union) if union else 0
            scored.append((score, i, doc))

        # 按得分降序排列，取 Top-K
        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for score, _, doc in scored[:top_k] if score > 0]

        # 拼接为上下文字符串
        context = "\n".join(f"- {doc}" for doc in top_docs)

        logger.debug(f"RAG 检索 | 查询: {query[:50]} | 命中: {len(top_docs)} 条")
        return context

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        简易中文分词（字符 bigram + 关键词提取）

        【设计决策】
        不引入 jieba 等重依赖，用 bigram 近似覆盖中文语义
        同时提取关键实体（B1、B2、充电、满位等）作为精确匹配信号

        【优化方向】
        生产环境替换为 jieba.cut() 或 sentence-transformers 编码
        """
        # 提取关键实体（硬编码高频词，确保精确命中）
        keywords = ["B1", "B2", "地下一层", "地下二层", "充电", "满位",
                    "空位", "导航", "路线", "收费", "A区", "B区", "入口"]
        tokens = [kw for kw in keywords if kw in text]

        # 字符 bigram（覆盖长尾语义）
        for i in range(len(text) - 1):
            tokens.append(text[i:i + 2])

        return tokens

    def reload(self):
        """
        热重载知识库（修改文件后无需重启服务）

        【调用方式】
        可通过管理接口 POST /api/admin/reload-knowledge 触发
        """
        self._load_knowledge()
        logger.info("RAG 知识库已热重载")


# ==================== 模块级单例 ====================
rag_service = RAGService()