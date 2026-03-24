# HW03: 人脸检测与识别 Web App (Streamlit)

本项目是一个基于 `face_recognition` 库和 `Streamlit` 框架开发的人脸检测与交互式识别 Web 应用。

## 1. 项目结构
```text
hw03/
├── app.py                 # Streamlit 前端交互与页面布局代码
├── face_utils.py          # 核心业务逻辑：人脸检测、128维特征编码与绘制
├── requirements.txt       # Python 依赖包清单
├── README.md              # 项目运行与说明文档
└── known_faces/           # (可选) 已知人脸底库目录
