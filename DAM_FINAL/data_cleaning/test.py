import argparse
import glob
import os
import time
import datetime
import numpy as np
import pandas as pd


dataframes = []
for index in range(2029, 2030):  # 这个循环会执行多次
    file_pattern = os.path.join('output/T2DM/csv/', f'{index}*.csv')
    all_files = glob.glob(file_pattern)

    # 打印文件名
    for file in all_files:
        filename = os.path.basename(file)
        print(filename)
        # 读取 CSV 文件并添加到 dataframes 列表中
        df_single = pd.read_csv(file)
        dataframes.append(df_single)

    # 将所有 DataFrame 合并为一个
df = pd.concat(dataframes, ignore_index=True)
print(df.shape)


df.to_csv("out.csv", index=False)

# 编码时间信息为数值特征
df[['Hour', 'Minute']] = df['Time'].str.split(':', expand=True)
df.drop(columns=['Time'], inplace=True)
df['Hour'] = pd.to_numeric(df['Hour'])
df['Minute'] = pd.to_numeric(df['Minute'])

# 保留特征列
feature_columns = ['Hour', 'Minute', "Gender (Female=1, Male=2)", "Age (years)", "Height (m)", "Weight (kg)",
                   "CSII - basal insulin (Novolin R, IU / H)"]
# 将特征列和目标列分开
feature_data = df[feature_columns].values
target_data = df[['CGM (mg / dl)']].values
print(target_data)

# 合并特征和目标数据，注意顺序
full_data = np.hstack((feature_data, target_data))

print("原始数据形状：", full_data.shape)
print("原始数据：", full_data[0])
# 计算并打印均值
mean_values = full_data.mean(axis=0)
print("均值：", mean_values)

# 计算并打印标准差
std_values = full_data.std(axis=0)
print("标准差：", std_values)


# # 标准化数据
# scaler = StandardScaler()
# scaler.fit(full_data)
# full_data_normalized = scaler.transform(torch.FloatTensor(full_data))
# print("标准化数据", full_data_normalized)