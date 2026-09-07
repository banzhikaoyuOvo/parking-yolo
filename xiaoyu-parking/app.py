"""
智能停车引导系统 - 主服务 (app.py)
====================================
【模块职责】
1. 大屏总览：随机抽取 B1/B2 各一张图片 → YOLO 检测 → 返回实时数据
2. 楼层详情：返回当前选定楼层的检测结果（与大屏一致）
3. AI 导航：RAG 检索 + LLM 流式生成导航话术（SSE）
4. 兼容接口：保留原有 /detect 上传检测接口

【架构流程】
用户打开终端
    → GET /api/screen/main（随机选图 + YOLO检测）
    → 用户点击某层
    → GET /api/floor/B1（返回缓存的检测结果）
    → POST /api/navigate（RAG + LLM 流式导航）

【设计决策】
- 车位数字 100% 来自 YOLO 检测，LLM 绝不参与数字生成（防幻觉）
- 大屏每次刷新重新随机选图 + 重新检测（模拟真实监控轮转）
- 用户点进详情时读取同一份检测结果（状态一致性）
- 导航话术由 LLM 生成，但路线知识来自 RAG 检索（防编造）

【维护须知】
- 新增 API 路由在本文件的 "API 路由" 区域添加
- 修改检测逻辑 → services/yolo_service.py
- 修改导航知识 → knowledge/parking_nav.txt
- 修改 LLM 行为 → config.py 中的 LLM_SYSTEM_PROMPT

【版本记录】
v4.0-yolo-rag: YOLO检测 + RAG检索 + LLM流式导航 三阶段流水线
v3.0-rag-isolation: user_id 隔离、429 限流重试
v1.0: 基础 YOLO + LLM
"""

import json
import os
import json
import random
import time
import uuid
import logging
import sys
import io
import base64
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS
from PIL import Image

from config import Config
from services.yolo_service import yolo_service
from services.rag_service import rag_service
from services.llm_service import stream_navigation, get_advice_safe

# ==================== 日志配置 ====================
# 【设计意图】结构化 JSON 日志，便于接入 ELK/Loki
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class JsonFormatter(logging.Formatter):
    """JSON 日志格式化器（扩展方式见 v3.0 注释）"""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        for key in ("empty_count", "user_id", "floor", "image"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        return json.dumps(log_data, ensure_ascii=False)


handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)


# ==================== Flask 应用初始化 ====================
app = Flask(__name__, static_folder=None)  # 禁用默认 static，手动管理
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH


# ==================== 画面状态管理 ====================
# 【设计决策】
# 大屏随机选图后，将"选了哪张图 + 检测结果"缓存到 _screen_state
# 用户点进详情时读取同一份缓存 → 保证"看到的图"和"看到的数字"一致
# 下次大屏刷新时重新随机 + 重新检测 → 覆盖旧状态
_screen_state: dict = {
    "B1": None,  # {"image": str, "detection": dict, "selected_at": float}
    "B2": None,
    "refresh_count": 0,
}


def _select_and_detect(floor_id: str) -> dict:
    """
    为指定楼层随机选取一张图片并执行 YOLO 检测

    【流程】
    1. 从 Config.CAMERA_POOL[floor_id]["images"] 中随机选一张
    2. 调用 yolo_service.detect_image() 获取车位数据
    3. 组装返回结构

    【返回值】
    {
        "floor_id": "B1",
        "floor_name": "地下一层",
        "image_file": "b1_50.jpg",
        "image_url": "/assets/cameras/b1_50.jpg",
        "total_spaces": 45,       # YOLO 检测到的车位总数
        "occupied": 22,
        "vacant": 23,
        "occupancy_rate": 48.9,
        "detections": [...],      # 完整检测框数据
        "selected_at": "17:26:03"
    }
    """
    pool = Config.CAMERA_POOL[floor_id]
    image_file = random.choice(pool["images"])
    image_path = str(Config.CAMERA_DIR / image_file)

    logger.info(f"随机选图 | 楼层: {floor_id} | 图片: {image_file}")

    # 执行 YOLO 检测
    detection = yolo_service.detect_image(image_path)

    return {
        "floor_id": floor_id,
        "floor_name": pool["name"],
        "image_file": image_file,
        "image_url": f"/assets/cameras/{image_file}",
        "total_spaces": detection["total_spaces"],
        "occupied": detection["occupied"],
        "vacant": detection["vacant"],
        "occupancy_rate": detection["occupancy_rate"],
        "detections": detection["detections"],
        "detection_error": detection["error"],
        "selected_at": datetime.now().strftime("%H:%M:%S"),
    }


# ==================== API 路由 ====================

@app.route("/api/screen/main", methods=["GET"])
def screen_main():
    """
    大屏总览接口：随机选图 + YOLO 检测

    【调用时机】前端每 15 秒轮询一次
    【核心逻辑】
    1. 为 B1、B2 各随机选一张图片
    2. 分别送入 YOLO 检测
    3. 缓存结果到 _screen_state（供详情页读取）
    4. 返回两层的检测数据

    【设计决策】
    每次刷新都重新检测（而非复用旧结果），模拟真实监控的"画面轮转"
    生产环境中可改为：摄像头推流 → 定时截帧 → 检测 → 推送
    """
    _screen_state["B1"] = _select_and_detect("B1")
    _screen_state["B2"] = _select_and_detect("B2")
    _screen_state["refresh_count"] += 1

    logger.info(
        f"大屏刷新 #{_screen_state['refresh_count']} | "
        f"B1: {_screen_state['B1']['image_file']} (空位{_screen_state['B1']['vacant']}) | "
        f"B2: {_screen_state['B2']['image_file']} (空位{_screen_state['B2']['vacant']})"
    )

    return jsonify({
        "code": 200,
        "data": {
            "B1": _screen_state["B1"],
            "B2": _screen_state["B2"],
        },
        "server_time": datetime.now().strftime("%H:%M:%S"),
        "refresh_count": _screen_state["refresh_count"],
    })


@app.route("/api/floor/<floor_id>", methods=["GET"])
def floor_detail(floor_id: str):
    """
    楼层详情接口：返回当前选定楼层的检测数据

    【设计决策】
    不重新随机/重新检测，直接读取 _screen_state 中的缓存
    保证用户看到的图片和数字与大屏完全一致（状态一致性）

    【边界处理】
    若大屏从未刷新过（_screen_state 为空），则立即执行一次选图+检测
    """
    fid = floor_id.upper()
    if fid not in Config.CAMERA_POOL:
        return jsonify({"code": 404, "message": f"未知楼层: {floor_id}"}), 404

    # 若缓存为空（首次访问），立即生成
    if _screen_state.get(fid) is None:
        _screen_state[fid] = _select_and_detect(fid)

    return jsonify({"code": 200, "data": _screen_state[fid]})


@app.route("/api/navigate", methods=["POST"])
def navigate():
    """
    AI 导航接口：RAG 检索 + LLM 流式生成（SSE）

    【请求体】JSON: {"floor": "B1"}
    【响应格式】SSE (Server-Sent Events)
        data: {"token": "收到"}
        data: {"token": "！B1"}
        data: {"token": "层目前"}
        ...
        data: [DONE]

    【三阶段流水线】
    1. 读取 YOLO 检测数据（来自 _screen_state 缓存）
    2. RAG 检索相关导航知识
    3. LLM 流式生成导航话术（数字由 YOLO 提供，路线由 RAG 提供）

    【设计决策】
    - 车位数字绝不经过 LLM 生成 → 杜绝"AI 编造车位数"
    - 导航路线来自 RAG 知识库 → 杜绝"AI 虚构路线"
    - LLM 只负责"把事实和知识组织成自然语言" → 最小化幻觉风险
    """
    data = request.get_json(silent=True)
    if not data or "floor" not in data:
        return jsonify({"code": 400, "message": "缺少 floor 参数"}), 400

    fid = data["floor"].upper()
    if fid not in Config.CAMERA_POOL:
        return jsonify({"code": 404, "message": f"未知楼层: {fid}"}), 404

    # 获取当前楼层的检测数据（若未检测过则立即执行）
    if _screen_state.get(fid) is None:
        _screen_state[fid] = _select_and_detect(fid)
    floor_data = _screen_state[fid]

    # 获取另一层的数据（用于满位时的替代建议）
    other_fid = "B2" if fid == "B1" else "B1"
    if _screen_state.get(other_fid) is None:
        _screen_state[other_fid] = _select_and_detect(other_fid)
    other_data = _screen_state[other_fid]

    # user_id 生成（设备隔离）
    device_id = request.headers.get("X-Device-ID", str(uuid.uuid4()))
    user_id = f"parking-nav-{device_id}"

    logger.info(
        f"导航请求 | 楼层: {fid} | 空位: {floor_data['vacant']} | user_id: {user_id}",
        extra={"floor": fid, "user_id": user_id}
    )

    def generate():
        """SSE 生成器：逐 token 推送 LLM 输出"""
        try:
            for token in stream_navigation(
                floor_name=floor_data["floor_name"],
                vacant=floor_data["vacant"],
                occupied=floor_data["occupied"],
                total=floor_data["total_spaces"],
                occupancy_rate=floor_data["occupancy_rate"],
                other_floor_name=other_data["floor_name"],
                other_vacant=other_data["vacant"],
                user_id=user_id,
            ):
                # SSE 协议：每条消息以 "data: " 开头，"\n\n" 结尾
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

            # 流结束标记
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"SSE 流异常: {e}")
            yield f"data: {json.dumps({'token': '导航服务异常，请稍后重试。'})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁止 Nginx 缓冲
            "Connection": "keep-alive",
        },
    )


# ==================== 兼容接口（保留原有 /detect） ====================

@app.route("/detect", methods=["POST"])
def detect():
    """
    【兼容接口】接收前端上传图片 → YOLO 检测 → LLM 建议

    【保留原因】向后兼容你原有的前端 Demo 页面
    【新流程】大屏终端使用 /api/screen/main + /api/navigate

    【请求体】JSON: {"image": "data:image/png;base64,..."}
    【响应体】JSON: {detections, total, occupied, vacant, advice}
    """
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "缺少 image 字段"}), 400

        image_b64 = data["image"]
        if image_b64.startswith("data:image"):
            image_b64 = image_b64.split(",")[1]

        # 解码 Base64 → 保存临时文件 → YOLO 检测
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 保存为临时文件（YOLO 服务接受文件路径）
        tmp_path = Config.CAMERA_DIR / f"_tmp_{uuid.uuid4().hex[:8]}.jpg"
        img.save(tmp_path)

        try:
            detection = yolo_service.detect_image(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)  # 清理临时文件

        # user_id
        device_id = request.headers.get("X-Device-ID", str(uuid.uuid4()))
        user_id = f"parking-device-{device_id}"

        # 调用 LLM 生成建议（非流式，兼容原有前端）
        advice = get_advice_safe(
            vacant=detection["vacant"],
            occupied=detection["occupied"],
            total=detection["total_spaces"],
            user_id=user_id,
        )

        return jsonify({
            "detections": detection["detections"],
            "total": detection["total_spaces"],
            "occupied": detection["occupied"],
            "vacant": detection["vacant"],
            "advice": advice,
        })

    except Exception as e:
        logger.error(f"/detect 接口异常: {e}")
        return jsonify({"error": str(e)}), 500


# ==================== 静态资源服务 ====================

@app.route("/assets/cameras/<filename>")
def serve_camera_image(filename: str):
    """
    提供停车场图片的静态访问

    【安全注意】使用 send_from_directory 防止路径穿越攻击
    """
    return send_from_directory(str(Config.CAMERA_DIR), filename)


@app.route("/")
def index():
    """前端大屏页面"""
    return send_from_directory(str(Config.BASE_DIR / "frontend"), "index.html")


# ==================== 管理接口 ====================

@app.route("/api/admin/reload-knowledge", methods=["POST"])
def reload_knowledge():
    """
    热重载 RAG 知识库（修改 parking_nav.txt 后无需重启）

    【使用场景】运营人员更新了导航规则后，调用此接口立即生效
    """
    rag_service.reload()
    return jsonify({"code": 200, "message": "知识库已重载"})


@app.route("/api/health")
def health():
    """健康检查（部署平台 / 负载均衡器用）"""
    return jsonify({
        "status": "ok",
        "service": "xiaoyu-parking",
        "version": "v4.0-yolo-rag",
        "yolo_loaded": yolo_service.model is not None,
        "rag_entries": len(rag_service.knowledge_base),
    })


# ==================== 启动入口 ====================
if __name__ == "__main__":
    logger.info(
        "智能停车引导系统启动",
        extra={"version": "v4.0-yolo-rag"}
    )
    # 【生产部署】debug=False + gunicorn -w 4 -b 0.0.0.0:5000 app:app
    app.run(host="0.0.0.0", port=5000, debug=True)