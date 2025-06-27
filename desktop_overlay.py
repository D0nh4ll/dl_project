import sys
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import mss
from PyQt5 import QtWidgets, QtGui, QtCore

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

class OverlayWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)  # 不可交互
        self.screen = QtWidgets.QApplication.primaryScreen()
        geometry = self.screen.geometry()
        self.setGeometry(geometry)
        self.setFixedSize(geometry.width(), geometry.height())
        self.detections = []
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(30)  # 约33ms一帧
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.frame_count = 0
        self.last_detections = []

    def update_overlay(self):
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        self.frame_count += 1
        if self.frame_count % 3 == 1:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image)
            detections = []
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
                    label = ''
                    if face_img.size != 0:
                        face_pil = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_pil = cv2.resize(face_pil, (224, 224))
                        face_pil = Image.fromarray(face_pil)
                        face_tensor = preprocess(face_pil).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = model(face_tensor)
                            pred = torch.argmax(output, 1).item()
                            label = class_names[pred]
                    detections.append((x1, y1, x2, y2, label))
            self.last_detections = detections
        self.detections = self.last_detections
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        for x1, y1, x2, y2, label in self.detections:
            # 半透明框
            color = QtGui.QColor(0, 255, 0, 120)
            painter.setBrush(color)
            painter.setPen(QtGui.QPen(QtGui.QColor(0,255,0,200), 3))
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            # 文字背景
            painter.setBrush(QtGui.QColor(0,0,0,160))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRect(x1, y1-30, 120, 30)
            # 文字
            painter.setPen(QtGui.QPen(QtGui.QColor(255,255,255,255)))
            font = QtGui.QFont()
            font.setPointSize(16)
            painter.setFont(font)
            painter.drawText(x1+5, y1-8, label)
        painter.end()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    overlay = OverlayWindow()
    overlay.show()
    sys.exit(app.exec_()) 