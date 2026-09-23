# 项目架构说明

> Parking-YOLO 的目录结构、数据流、模块职责

## 启动方式

- 浏览器打开：http://localhost:5000
- 终端启动：python app.py

## 目录结构

parking-yolo/
├── app.py                       # Flask 主服务 + API 路由
├── config.py                    # 配置中心
├── requirements.txt             # 依赖
│
├── services/                    # 服务层（单例）
│   ├── __init__.py
│   ├── yolo_service.py          # YOLOv8 检测服务
│   ├── rag_service.py           # RAG 检索服务
│   └── llm_service.py           # LLM 生成服务（含流式）
│
├── assets/
│   └── cameras/                 # 模拟摄像头画面
│       ├── b1_empty.jpg         # B1-01 空停车场
│       ├── b1_20.jpg            # B1-02 占用 20%
│       ├── b1_50.jpg            # B1-03 占用 50%
│       ├── b1_90.jpg            # B1-04 占用 90%
│       ├── b2_empty.jpg         # B2-01 空停车场
│       ├── b2_20.jpg            # B2-02 占用 20%
│       ├── b2_50.jpg            # B2-03 占用 50%
│       └── b2_full.jpg          # B2-04 占用 100%
│
├── knowledge/
│   └── parking_nav.txt          # RAG 知识库（导航规则）
│
├── frontend/
│   └── index.html               # 监控终端大屏
│
├── models/                      # YOLO 权重（不进 Git，从 Release 下载）
│   └── best.pt
│
├── training/
│   └── parking_data.yaml        # YOLOv8 训练数据集配置
│
└── docs/                        # 项目文档
    ├── ARCHITECTURE.md          # 本文件
    └── screenshots/             # 效果截图

## 数据流

用户打开终端
    |
    v
GET /api/screen/main --> 随机选 2 张图 --> YOLOv8 检测 --> 返回车位数据
    |                                                        |
    v                                                        v
大屏显示监控画面 + 实时数字                          缓存到 _screen_state
    |
    v 用户点击"前往B1层"
    |
GET /api/floor/B1 --> 读取缓存（同一张图、同一组数字）
    |
    v
POST /api/navigate --> RAG 检索导航知识 --> LLM 流式生成话术 --> SSE 推送
                         |                        |
                         |                        +-- 数字来自 YOLO（事实）
                         +-- 路线来自知识库（约束）    LLM 只负责"说话"
