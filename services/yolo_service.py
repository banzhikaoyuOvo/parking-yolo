"""
YOLOv8 车位检测服务 (yolo_service.py)
======================================
【模块职责】
封装 YOLOv8 模型的加载、推理、结果解析逻辑
对外提供统一的 detect_image() 接口，屏蔽模型细节

【设计决策】
- 模型在模块加载时初始化（单例），避免每次请求重复加载权重文件
- 检测结果只返回结构化数据（dict），不返回原始 tensor
- 置信度过滤在推理阶段完成（YOLO conf 参数），减少后处理开销

【维护须知】
- 更换模型文件时，只需修改 config.py 中的 YOLO_MODEL_PATH
- 若类别数量变化，需同步更新 config.py 中的 YOLO_CLASS_NAMES
- 本模块不处理 HTTP 请求，纯粹是"图片进 → 数据出"的服务层

【版本记录】
v4.0: 从 app.py 中抽离为独立服务模块，支持批量检测
v1.0: 内嵌在 app.py 中的检测逻辑
"""

import logging
from pathlib import Path
from typing import Optional

from PIL import Image
from ultralytics import YOLO

from config import Config

logger = logging.getLogger(__name__)


class YOLOService:
    """
    YOLOv8 车位检测服务（单例模式）

    【使用方式】
        from services.yolo_service import yolo_service
        result = yolo_service.detect_image("path/to/image.jpg")

    【线程安全说明】
    YOLO 推理本身是线程安全的（PyTorch 内部有锁），
    但多并发场景下建议使用模型池（见优化方向）
    """

    def __init__(self):
        """
        初始化 YOLO 模型
        【性能注意】模型加载约需 2-5 秒，仅在服务启动时执行一次
        """
        logger.info(f"正在加载 YOLO 模型: {Config.YOLO_MODEL_PATH}")
        self.model = YOLO(Config.YOLO_MODEL_PATH)
        self.class_names = Config.YOLO_CLASS_NAMES
        self.conf_threshold = Config.YOLO_CONFIDENCE_THRESHOLD
        logger.info(
            f"YOLO 模型加载完成 | 类别: {self.class_names} | 置信度阈值: {self.conf_threshold}"
        )

    def detect_image(self, image_path: str) -> dict:
        """
        对单张图片执行车位检测

        【参数】
        - image_path: 图片文件的绝对/相对路径

        【返回值】
        {
            "total_spaces": int,      # 检测到的车位总数
            "occupied": int,          # 已占用车位数
            "vacant": int,            # 空闲车位数
            "occupancy_rate": float,  # 占用率 (0~100)
            "detections": [           # 每个检测框的详情
                {"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}
            ]
        }

        【异常处理】
        - 图片不存在 → 返回全零结果 + 错误标记
        - 模型推理异常 → 向上抛出，由调用方兜底
        """
        # 防御性检查：文件是否存在
        if not Path(image_path).exists():
            logger.error(f"图片文件不存在: {image_path}")
            return self._empty_result(error=f"文件不存在: {image_path}")

        try:
            # 读取图片并执行推理
            img = Image.open(image_path).convert("RGB")

            # 【关键参数】conf=置信度阈值，verbose=False 抑制 YOLO 的控制台输出
            results = self.model(img, conf=self.conf_threshold, verbose=False)

            # 解析检测框
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = box.conf[0].item()
                    xyxy = box.xyxy[0].tolist()

                    # 防御性编程：防止模型输出超出类别列表范围的 ID
                    class_name = (
                        self.class_names[cls_id]
                        if cls_id < len(self.class_names)
                        else "unknown"
                    )
                    detections.append({
                        "class": class_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(v, 1) for v in xyxy],
                    })

            # 统计车位状态
            # 【设计决策】只统计 space-empty 和 space-occupied 两类
            # 'spaces' 类别是整体区域标注，不参与计数
            occupied = sum(1 for d in detections if d["class"] == "space-occupied")
            vacant = sum(1 for d in detections if d["class"] == "space-empty")
            total = occupied + vacant

            # 计算占用率（防止除零）
            occupancy_rate = round((occupied / total * 100), 1) if total > 0 else 0.0

            logger.info(
                f"检测完成 | 图片: {Path(image_path).name} | "
                f"总车位: {total} | 占用: {occupied} | 空闲: {vacant} | "
                f"占用率: {occupancy_rate}%"
            )

            return {
                "total_spaces": total,
                "occupied": occupied,
                "vacant": vacant,
                "occupancy_rate": occupancy_rate,
                "detections": detections,
                "error": None,
            }

        except Exception as e:
            logger.error(f"YOLO 推理异常 | 图片: {image_path} | 错误: {e}")
            return self._empty_result(error=str(e))

    def detect_batch(self, image_paths: list[str]) -> list[dict]:
        """
        批量检测多张图片（当前为串行，后续可改为并行）

        【参数】
        - image_paths: 图片路径列表

        【返回值】
        - 与 detect_image 返回值相同的列表

        【优化方向】
        当图片数量 > 4 时，可考虑使用 ThreadPoolExecutor 并行推理
        或使用 YOLO 原生的 batch 推理模式（需统一图片尺寸）
        """
        return [self.detect_image(p) for p in image_paths]

    @staticmethod
    def _empty_result(error: Optional[str] = None) -> dict:
        """返回空结果（用于异常兜底）"""
        return {
            "total_spaces": 0,
            "occupied": 0,
            "vacant": 0,
            "occupancy_rate": 0.0,
            "detections": [],
            "error": error,
        }


# ==================== 模块级单例 ====================
# 【设计决策】在 import 时即加载模型，后续所有请求共享同一实例
# 避免每次 HTTP 请求都重新加载 50MB+ 的权重文件
yolo_service = YOLOService()