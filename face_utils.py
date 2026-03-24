
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def load_known_faces(known_faces_dir="known_faces"):
    """
    加载已知人脸库（可选功能）
    遍历目录下的图片，提取特征编码并与文件名（去掉后缀）作为人名绑定
    """
    known_encodings = []
    known_names = []
    
    if not os.path.exists(known_faces_dir):
        return known_encodings, known_names

    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(filepath)
            # 获取图片中的第一个人脸编码
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                # 以文件名作为人名 (例如: "张三.jpg" -> "张三")
                known_names.append(os.path.splitext(filename)[0])
                
    return known_encodings, known_names

def process_and_draw_faces(uploaded_file, known_encodings, known_names):
    """
    处理上传的图片，检测人脸、识别身份并在图像上画框和标签
    """
    # 1. 加载图像并转换为 numpy 数组
    img_image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(img_image)

    # 2. 检测人脸位置和特征编码 (128维)
    face_locations = face_recognition.face_locations(img_array)
    face_encodings = face_recognition.face_encodings(img_array, face_locations)

    # 3. 使用 Pillow 画框
    draw = ImageDraw.Draw(img_image)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        
        # 如果有已知人脸库，进行比对识别
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

        # 画矩形框 (红色)
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=3)
        
        # 画标签底框和文字
        text_bbox = draw.textbbox((left, bottom), name)
        draw.rectangle(((left, bottom), (right, text_bbox[3] + 5)), fill=(255, 0, 0), outline=(255, 0, 0))
        draw.text((left + 6, bottom + 2), name, fill=(255, 255, 255))

    return img_image, len(face_locations)
