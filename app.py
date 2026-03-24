import streamlit as st
from face_utils import load_known_faces, process_and_draw_faces

# 设置页面配置
st.set_page_config(page_title="人脸检测与识别 Web App", page_icon="👤", layout="centered")

st.title("👤 人脸检测与识别系统")
st.markdown("基于 `face_recognition` 和 `Streamlit` 搭建。上传一张图片，系统会自动检测人脸位置。")

# 加载已知人脸库 (利用缓存避免重复加载)
@st.cache_resource
def get_known_faces():
    return load_known_faces("known_faces")

known_encodings, known_names = get_known_faces()

if known_names:
    st.sidebar.success(f"已加载人脸库：{', '.join(known_names)}")
else:
    st.sidebar.info("未检测到已知人脸库。目前仅工作在'人脸检测'模式。若需识别，请在 `known_faces/` 目录下放入人脸图片。")

# 文件上传组件
uploaded_file = st.file_uploader("请选择一张图片上传 (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示原始图片
    st.subheader("原始图片")
    st.image(uploaded_file, caption="上传的图片", use_column_width=True)
    
    # 增加一个处理按钮，提升交互感
    if st.button("开始检测 / 识别", type="primary"):
        with st.spinner("正在使用深度学习模型进行面部特征提取与比对，请稍候..."):
            try:
                # 调用核心处理函数
                result_img, face_count = process_and_draw_faces(uploaded_file, known_encodings, known_names)
                
                # 展示结果
                st.success(f"处理完成！共检测到 {face_count} 张人脸。")
                st.subheader("检测结果")
                st.image(result_img, caption="带有人脸框和标签的结果图", use_column_width=True)
            except Exception as e:
                st.error(f"处理图片时发生错误: {e}")
