import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image

# 配置
model_path = 'best_resnet18.pth'
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
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
        cv2.imshow('Face Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows() 