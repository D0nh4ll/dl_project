import os
import matplotlib.pyplot as plt
from PIL import Image
import random

train_dir = 'train'
classes = sorted(os.listdir(train_dir))
counts = []
sample_imgs = []

for cls in classes:
    cls_dir = os.path.join(train_dir, cls)
    img_list = [f for f in os.listdir(cls_dir) if f.lower().endswith(('jpg','jpeg','png','bmp'))]
    counts.append(len(img_list))
    if img_list:
        sample_imgs.append(os.path.join(cls_dir, random.choice(img_list)))
    else:
        sample_imgs.append(None)

# 绘制类别分布直方图
plt.figure(figsize=(10,4))
plt.bar(classes, counts, color='skyblue')
plt.xlabel('Class')
plt.ylabel('Image Count')
plt.title('Training Set Class Distribution')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

# 展示每类一张样本图片
plt.figure(figsize=(14, 2.5))
for i, (cls, img_path) in enumerate(zip(classes, sample_imgs)):
    plt.subplot(1, len(classes), i+1)
    if img_path:
        img = Image.open(img_path)
        plt.imshow(img)
    plt.title(cls)
    plt.axis('off')
plt.tight_layout()
plt.savefig('class_samples.png')
plt.show() 