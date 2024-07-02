import warnings

import pandas as pd
import glob
import os
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

for year in range(2000, 2100):
    file_pattern = os.path.join('generated_data/Shanghai_T2DM/', f'{year}*')
    all_files = glob.glob(file_pattern)
    # 提取文件名
    all_file_names = [os.path.basename(file) for file in all_files]
    # 打印文件名
    # print(all_file_names)

    # # 匹配模式
    # file_pattern = os.path.join('data/Shanghai_T2DM/', '2008*')
    # all_files = glob.glob(file_pattern)
    #
    # # 提取文件名
    # all_file_names = [os.path.basename(file) for file in all_files]
    #
    # # 打印文件名
    # print(all_file_names)

    for filename in all_file_names:
        # 匹配的名字没有文件后缀
        Match_name = filename.replace('.xlsx', '').replace('.xls', '')

        # 从总表中读取包含要匹配数据的表
        summary_data_table = pd.read_excel('generated_data/Shanghai_T2DM_Summary.xlsx')
        # 根据文件名从总表中找到匹配行
        filtered_dt = summary_data_table[summary_data_table['Patient Number'].str.contains(Match_name)]
        # 提取前五列（ID，性别，年龄，身高，体重）
        summary_needed_columns = filtered_dt.iloc[:, :5]

        # 将提取数据和对应的病人的表合并
        patient_data_table = pd.read_excel('generated_data/Shanghai_T2DM/' + filename)

        patient_data_table['Date'] = pd.to_datetime(patient_data_table['Date'])

        # 提取时间部分并转换为字符串
        patient_data_table['Time'] = patient_data_table['Date'].dt.strftime('%H:%M')

        # 删除原始 'Date' 列
        patient_data_table.drop(columns=['Date'], inplace=True)
        Time_column = patient_data_table.pop('Time')
        patient_data_table.insert(0, 'Time', Time_column)

        # 提取病人表中时间、CGM（连续血糖监测）、CSII - basal insulin (Novolin R  IU / H)（皮下注射胰岛素剂量）.
        index = 11
        patient_needed_columns = patient_data_table.iloc[:, [0, 1, index]]
        # 获取第三列的列索引
        column_index = 2
        patient_needed_columns.iloc[:, column_index] = patient_needed_columns.iloc[:, column_index].fillna(0)

        # print(patient_needed_columns)

        # 将 summary_needed_columns 重复拼接到与 patient_data_table 行数相匹配
        summary_needed_columns_repeated = pd.concat([summary_needed_columns] * len(patient_needed_columns),
                                                    ignore_index=True)

        # 将 patient_data_table 和 summary_needed_columns_repeated 拼接到一起
        result = pd.concat([summary_needed_columns_repeated, patient_needed_columns], axis=1)
        result = result.drop(result.columns[0], axis=1)
        first_column = result.pop('Time')
        result.insert(0, 'Time', first_column)

        if result.columns[6] != 'CSII - basal insulin (Novolin R, IU / H)':
            result.columns.values[6] = 'CSII - basal insulin (Novolin R, IU / H)'
        if result.columns[5].startswith('CGM'):
            result.columns.values[5] = 'CGM (mg / dl)'

        print('输出名字：' + Match_name)
        # 将 result 存储为 Excel 表格
        output_file = 'output/T2DM/excel/' + Match_name + '.xlsx'
        result.to_excel(output_file, index=False)  # 如果不想要保存索引，可以设置 index=False

        # 将 result 存储为csv表格
        output_file = 'output/T2DM/csv/' + Match_name + '.csv'
        result.to_csv(output_file, index=False)  # 如果不想要保存索引，可以设置 index=False
