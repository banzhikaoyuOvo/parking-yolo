"""
LLM 导航生成服务 (llm_service.py)
==================================
【模块职责】
封装 DeepSeek LLM 的调用逻辑，提供：
1. 非流式调用（get_advice_safe）—— 用于 /detect 接口
2. 流式调用（stream_navigation）—— 用于 /api/navigate SSE 接口

【设计决策】
- 车位数字由 YOLO 提供，作为"事实"注入提示词，LLM 只负责"表达"
- RAG 检索结果作为"知识约束"注入，防止 LLM 编造路线
- 流式输出使用 SSE 协议，前端逐字渲染，体验接近 ChatGPT

【维护须知】
- 所有 LLM 调用必须通过本模块的 safe/stream 入口
- 禁止在业务代码中直接 import OpenAI 客户端
- user_id 格式：[a-zA-Z0-9\\-_]+，最大 512 字符

【版本记录】
v4.0: 新增流式输出、RAG 上下文注入
v3.0: 新增 user_id 隔离、429 限流重试
v1.0: 基础非流式调用
"""

import json
import re
import time
import logging
from typing import Generator

from openai import OpenAI, RateLimitError

from config import Config, LLM_SYSTEM_PROMPT
from services.rag_service import rag_service

logger = logging.getLogger(__name__)

# ==================== LLM 客户端（全局单例） ====================
# 【注意】避免每次请求重复创建 TCP 连接
llm_client = OpenAI(
    api_key=Config.DEEPSEEK_API_KEY,
    base_url=Config.LLM_BASE_URL,
)


def _build_navigation_prompt(
    floor_name: str,
    vacant: int,
    occupied: int,
    total: int,
    occupancy_rate: float,
    other_floor_name: str,
    other_vacant: int,
) -> str:
    """
    构建导航提示词（将 YOLO 数据 + RAG 知识组装为 LLM 输入）

    【设计决策】
    - 数字部分用【】括起来，明确告知 LLM "这是事实，不要改"
    - RAG 知识用 <knowledge> 标签包裹，与用户指令区分
    - 另一层的数据也注入，让 LLM 在满位时能给出替代建议

    【参数】全部来自 YOLO 检测结果和 RAG 检索
    """
    # 第一步：RAG 检索相关知识
    query = f"{floor_name} 停车导航 路线 空位"
    rag_context = rag_service.retrieve(query, top_k=Config.RAG_TOP_K)

    # 第二步：组装提示词
    prompt = f"""<knowledge>
{rag_context if rag_context else "暂无相关导航知识"}
</knowledge>

<realtime_data>
【当前楼层】{floor_name}
【空闲车位】{vacant} 个
【已占用车位】{occupied} 个
【车位总数】{total} 个
【占用率】{occupancy_rate}%
【另一层（{other_floor_name}）空闲】{other_vacant} 个
</realtime_data>

请根据以上实时数据和导航知识，为车主生成一段停车引导话术。
要求：
1. 车位数字必须使用上述数据，禁止修改
2. 导航路线必须基于 knowledge 中的规则
3. 如果当前层占用率超过90%，建议车主考虑另一层
4. 语气亲切，150字以内
"""
    return prompt


def stream_navigation(
    floor_name: str,
    vacant: int,
    occupied: int,
    total: int,
    occupancy_rate: float,
    other_floor_name: str,
    other_vacant: int,
    user_id: str,
) -> Generator[str, None, None]:
    """
    流式生成导航话术（SSE 逐 token 输出）

    【调用方式】
        for chunk in stream_navigation(...):
            yield f"data: {json.dumps({'token': chunk})}\n\n"

    【参数】
    - floor_name ~ other_vacant: YOLO 检测数据（事实）
    - user_id: 设备标识，用于 API 隔离

    【返回值】
    - Generator[str]: 每次 yield 一小段文本（1~3个字）

    【异常处理】
    - RateLimitError: 指数退避重试（最多3次）
    - 其他异常: yield 兜底文案，不中断 SSE 流
    """
    prompt = _build_navigation_prompt(
        floor_name, vacant, occupied, total,
        occupancy_rate, other_floor_name, other_vacant
    )

    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            # 【关键】stream=True 开启流式返回
            response = llm_client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                # DeepSeek user_id 隔离（内容安全 + KVCache 隐私）
                extra_body={"user_id": user_id},
            )

            # 逐 chunk 读取流式响应
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
            return  # 正常结束

        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(
                    f"LLM 限流(429)，{wait_time}s 后重试 | user_id: {user_id}"
                )
                time.sleep(wait_time)
            else:
                logger.error(f"LLM 限流重试耗尽 | user_id: {user_id}")
                yield "抱歉，导航员当前繁忙，请稍后再试。您可以参考屏幕上的车位统计自行选择。"
                return

        except Exception as e:
            logger.error(f"LLM 流式调用异常: {e} | user_id: {user_id}")
            yield f"导航服务暂时不可用（{type(e).__name__}），请参考屏幕上的实时车位数据。"
            return


def get_advice_safe(
    vacant: int,
    occupied: int,
    total: int,
    user_id: str,
) -> dict:
    """
    非流式 LLM 调用（兼容原有 /detect 接口）

    【返回值】
    - dict: {"level": "abundant|tight|full|error", "suggestion": "..."}

    【设计决策】
    保留此函数是为了向后兼容你原有的 /detect 接口
    新流程推荐使用 stream_navigation()
    """
    fallback = {
        "level": "error",
        "suggestion": "导航员暂时走神了，请查看屏幕上的车位统计哦~"
    }

    # 根据占用率确定 level（这个判断不依赖 LLM，纯规则）
    if total == 0:
        return {"level": "error", "suggestion": "未检测到车位数据"}
    occupancy = occupied / total
    if occupancy < 0.5:
        level = "abundant"
    elif occupancy < 0.9:
        level = "tight"
    else:
        level = "full"

    try:
        prompt = f"当前空位 {vacant} 个，占用 {occupied} 个，总计 {total} 个。请给出简短停车建议。"
        response = llm_client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            extra_body={"user_id": user_id},
        )
        suggestion = response.choices[0].message.content.strip()
        return {"level": level, "suggestion": suggestion}

    except Exception as e:
        logger.error(f"LLM 非流式调用失败: {e} | user_id: {user_id}")
        return fallback