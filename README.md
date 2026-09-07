# 🚗 Parking-YOLO: 智能停车场视觉感知与导航系统

> An end-to-end AI application prototype integrating YOLOv8, RAG, and LLM for real-time visual perception, knowledge retrieval, and streaming spatial navigation.

## 📖 项目简介

本项目是一个基于计算机视觉和大语言模型的智能停车辅助系统。通过 YOLOv8 实时检测车位占用状态，结合 RAG 检索增强生成技术，为用户提供自然语言交互的空间导航服务。

**核心亮点：**
- **YOLOv8 实时车位检测**：精准识别空闲/占用车位，输出占用率数据
- **RAG + LLM 智能问答**：基于停车场知识库，生成个性化导航话术
- **SSE 流式推送**：LLM 实时流式输出生成内容，用户体验丝滑
- **Web 监控大屏**：多路摄像头画面实时展示，一键选择目标楼层

## 🛠️ 技术栈

| 模块 | 技术 |
|------|------|
| **AI 模型** | YOLOv8 (Object Detection) |
| **后端框架** | Flask |
| **LLM/RAG** | 向量数据库 + 流式生成 (SSE) |
| **前端** | HTML5 / CSS3 / JavaScript (原生) |
| **字体** | Google Fonts (Noto Sans SC, Rajdhani, Share Tech Mono) |
| **环境** | Python 3.10+ |

## 📁 项目结构

```text
xiaoyu-parking/
├── config.py                    # 配置中心
├── app.py                       # Flask 主服务
├── services/
│   ├── yolo_service.py          # YOLOv8 检测服务
│   ├── rag_service.py           # RAG 检索服务
│   └── llm_service.py           # LLM 生成服务（含流式）
├── assets/
│   └── cameras/                 # 模拟摄像头画面素材
├── knowledge/
│   └── parking_nav.txt          # RAG 知识库（导航规则、收费标准等）
├── frontend/
│   └── index.html               # 监控终端大屏页面
└── requirements.txt             # 依赖列表

🚀 快速开始
1. 克隆项目
git clone https://github.com/banzhikaoyuOvo/parking-yolo.git
cd xiaoyu-parking
🧠核心逻辑说明
系统采用"事实与表达分离"的设计模式：
1.视觉感知层：YOLOv8 负责从摄像头画面中提取客观事实（如车位占用率、空位数量）。
2.知识检索层：RAG 服务根据用户意图检索知识库中的导航规则、收费标准等约束信息。
3.语言生成层：LLM 将"视觉事实"与"导航规则"融合，生成自然、流畅的引导话术，并通过 SSE 流式推送到前端。
