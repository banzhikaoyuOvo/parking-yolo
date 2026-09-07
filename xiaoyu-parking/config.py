"""
智能停车引导系统 - 配置中心 (config.py)
========================================
【模块职责】
集中管理所有可变配置项，禁止在业务代码中硬编码

【维护须知】
- 新增配置项必须在此文件定义，并附带注释说明用途和取值范围
- 敏感信息（API Key）通过环境变量注入，禁止提交到版本控制
- 修改模型路径后，需同步验证 class_names 是否匹配

【版本记录】
v4.0: 新增 YOLO 图片池配置、RAG 知识库路径、流式输出参数
v3.0: 新增 user_id 隔离、限流重试配置
v1.0: 基础 LLM 配置
"""

import os
from pathlib import Path

# 项目根目录（所有相对路径基于此计算）
BASE_DIR = Path(__file__).parent


class Config:
    """全局配置类"""
    BASE_DIR: Path = BASE_DIR
    # ==================== LLM 配置 ====================
    # 【注意】API Key 从环境变量读取，本地开发可在 .env 文件中设置
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "sk-your-key-here")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    LLM_MODEL: str = "deepseek-chat"

    # ==================== YOLO 模型配置 ====================
    # ⚠️ 更换模型时必须同步更新 YOLO_CLASS_NAMES
    YOLO_MODEL_PATH: str = str(BASE_DIR / "models" / "best.pt")

    # 类别映射（索引必须与训练时 classes.txt 严格对应）
    # 【扩展方式】新增类别时在列表末尾追加，不要改变已有索引顺序
    YOLO_CLASS_NAMES: list = ['spaces', 'space-empty', 'space-occupied']

    # YOLO 推理置信度阈值（低于此值的检测框将被过滤）
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5

    # ==================== 图片池配置 ====================
    # 【设计决策】8张图片按楼层分组，每次刷新随机抽取各1张
    # 图片命名规则：{楼层}_{占用率描述}.jpg
    CAMERA_POOL: dict = {
        "B1": {
            "name": "地下一层",
            "total_spaces": 120,  # 该层物理车位总数（用于计算百分比）
            "images": [
                "b1_empty.jpg",   # 车位全空
                "b1_20.jpg",      # 占用约20%
                "b1_50.jpg",      # 占用约50%
                "b1_90.jpg",      # 占用约90%
            ],
        },
        "B2": {
            "name": "地下二层",
            "total_spaces": 80,
            "images": [
                "b2_empty.jpg",
                "b2_20.jpg",
                "b2_50.jpg",
                "b2_full.jpg",    # 车位全满
            ],
        },
    }

    # 图片存储目录
    CAMERA_DIR: Path = BASE_DIR / "assets" / "cameras"

    # ==================== RAG 知识库配置 ====================
    # 知识库文件路径（纯文本，每行一条规则）
    RAG_KNOWLEDGE_PATH: Path = BASE_DIR / "knowledge" / "parking_nav.txt"

    # RAG 检索返回的最大上下文条数
    RAG_TOP_K: int = 3

    # ==================== 流式输出配置 ====================
    # SSE 每次推送的字符数（模拟逐 token 输出）
    STREAM_CHUNK_SIZE: int = 2
    # 每次推送的间隔（秒），越小越快
    STREAM_INTERVAL: float = 0.03

    # ==================== 服务配置 ====================
    # 大屏自动刷新间隔（秒）
    SCREEN_REFRESH_INTERVAL: int = 15
    # 最大请求体大小（防止超大图片撑爆内存）
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024


# ==================== LLM 系统提示词 ====================
# 【设计决策】将检测数据作为"事实"注入，LLM 只负责"润色表达"
# 这样即使 LLM 产生幻觉，核心数字仍然正确（数字由 YOLO 提供）
LLM_SYSTEM_PROMPT = """你是"小鱼"智能停车场的AI导航员，负责为车主提供停车引导。

【你的职责】
1. 根据提供的实时车位检测数据，给出停车建议
2. 结合知识库中的场内导航规则，指引车主到达具体区域
3. 语气亲切专业，像一位经验丰富的停车场管理员

【严格约束】
- 车位数量必须使用提供的检测数据，禁止自行编造数字
- 如果某层车位已满，必须建议车主前往另一层
- 导航路线必须基于知识库中的规则，禁止虚构路线
- 回复控制在150字以内，简洁实用

【输出格式】
直接输出自然语言导航文本，不要输出JSON，不要输出markdown标记。
"""

# ==================== RAG 导航知识库内容 ====================
# 当 knowledge/parking_nav.txt 不存在时，使用此内置知识
RAG_FALLBACK_KNOWLEDGE = """
B1层导航规则：从主入口匝道下行至地下一层，下坡后限速5km/h。右转沿蓝色导引标线行驶约50米到达A区。A区空位充足时优先推荐。继续直行100米到达B区。B区靠近电梯口，适合需要快速上楼的车主。
B2层导航规则：从主入口匝道下行，经过B1层后继续下行至地下二层。沿黄色标线行驶。A区在左侧，有20个新能源充电桩。B区在右侧，靠近货梯。
收费规则：前30分钟免费。之后每小时5元，每日封顶30元。新能源车充电期间停车费减半。
满位处理：当某层车位占用率超过95%时，引导屏将显示红色满位标志。此时应建议车主前往另一层。
安全提示：场内限速5km/h。倒车入库时注意后方行人。停好后请车头朝外，便于紧急疏散。锁好车窗，贵重物品随身携带。
"""