


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 读取 CSV 文件
df = pd.read_csv('statistics/srf_4_train_results.csv')

# 提取需要的数据
x = df['Epoch']  # 横轴数据
y = df['PSNR']  # 纵轴数据

# 绘制折线图
plt.plot(x, y)
plt.xlabel('Epoch')  # 设置横轴标签
plt.ylabel('PSNR')  # 设置纵轴标签
plt.title('PSNR vs Epoch')  # 设置图标题
# 显示网格
plt.show()  # 显示图形

