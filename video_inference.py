import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
from tkinter import Tk, filedialog
from tqdm import tqdm

# 配置
model_path = 'best_resnet18.pth'
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)

# 文件选择
Tk().withdraw()
video_path = filedialog.askopenfilename(title='选择视频文件', filetypes=[('Video Files', '*.mp4;*.avi;*.mov;*.mkv')])
if not video_path:
    print('未选择视频文件，程序退出。')
    exit()

# 加载模型
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# mediapipe初始化
mp_face_detection = mp.solutions.face_detection

# 打开视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('无法打开视频文件')
    exit()

# 输出视频设置
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = os.path.join(output_dir, f"result_{os.path.basename(video_path)}")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 获取总帧数
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    pbar = tqdm(total=frame_count, desc='Processing Video')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x1 + w), min(ih, y1 + h)
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                # 预处理
                face_pil = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_pil = cv2.resize(face_pil, (224, 224))
                face_pil = Image.fromarray(face_pil)
                face_tensor = preprocess(face_pil).unsqueeze(0).to(device)
                # 推理
                with torch.no_grad():
                    output = model(face_tensor)
                    pred = torch.argmax(output, 1).item()
                    label = class_names[pred]
                # 画框和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        out.write(frame)
        pbar.update(1)
    pbar.close()
cap.release()
out.release()
print(f'处理完成，结果已保存到 {output_path}') 