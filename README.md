# 面部表情识别项目

## 项目简介
本项目基于 PyTorch、MediaPipe 和 ResNet50，实现了高精度的人脸表情识别。支持摄像头实时分析、视频文件批量处理、屏幕内容实时分析，并具备丰富的数据增强和可视化分析能力。

## 环境依赖
- Python 3.8+
- torch
- torchvision
- opencv-python
- mediapipe
- tqdm
- pillow
- albumentations
- mss
- PyQt5
- matplotlib
- pandas

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据集结构
项目根目录下应有如下结构：
```
train/
  angry/
  disgusted/
  fearful/
  happy/
  neutral/
  sad/
  surprised/
test/
  angry/
  disgusted/
  fearful/
  happy/
  neutral/
  sad/
  surprised/
```
每个类别文件夹下为对应表情的图片。

## 数据可视化
- 运行 `plot_dataset.py` 可生成训练集类别分布直方图（class_distribution.png）和每类样本示例（class_samples.png）。

## 训练模型
- 主干网络：ResNet50（ImageNet预训练）
- 数据增强：Albumentations（随机裁剪、旋转、亮度、遮挡等）
- 每个epoch每类随机采样400张图片，提升均衡性和泛化能力
- 损失函数：带label smoothing的交叉熵
- 优化器：AdamW，学习率调度CosineAnnealingLR
- 早停机制防止过拟合
- 训练曲线和精度自动保存为 train_history.csv

运行训练：
```bash
python train.py
```
训练完成后会在当前目录下生成 `best_resnet50.pth` 和 `train_history.csv`。

## 训练过程可视化
- 运行 `plot_history.py` 可生成训练loss和验证精度曲线（train_curve.png）。

## 摄像头实时表情识别
运行：
```bash
python camera_recog.py
```
摄像头画面中检测到人脸会实时显示表情类别。

## 视频文件批量表情识别
运行：
```bash
python video_inference.py
```
后台处理视频，终端显示进度，结果视频保存在 `output/` 目录。

## 屏幕内容实时分析与桌面叠加
- 推荐使用 `desktop_overlay.py`，该脚本基于 PyQt5 实现全屏透明无边框窗口，支持不可交互的桌面实时表情叠加，检测结果直接覆盖在主屏幕上，适合演示和实际应用。

## 结果展示
- 支持多目标（多个人脸）检测与识别。
- 训练过程和推理过程均有进度条和可视化。
- 可视化脚本辅助分析数据分布和模型表现。


