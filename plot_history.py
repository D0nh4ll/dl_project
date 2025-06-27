import pandas as pd
import matplotlib.pyplot as plt

# 读取训练历史
history = pd.read_csv('train_history.csv')

# 绘制曲线
plt.figure(figsize=(10,5))
plt.plot(history['epoch'], history['loss'], label='Loss', marker='o')
plt.plot(history['epoch'], history['val_acc'], label='Val Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('train_curve.png')
plt.show() 