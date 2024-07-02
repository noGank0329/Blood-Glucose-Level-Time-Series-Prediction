## README

### 项目概述
以T2DM为例，T1DM的操作与T2DM的操作完全一致

本项目包括两个主要脚本，`pre_T2DM.py` 和 `T2DM.py`，用于处理和转换上海T2DM（糖尿病）数据集。这些脚本旨在提取、处理和转换数据，以便进一步的分析和建模使用。

### 环境配置

在开始使用本项目之前，请确保你的环境已经安装了以下依赖包：

- Python 3.6+
- pandas
- openpyxl
- glob
- os
- warnings

你可以通过以下命令安装这些依赖：

```bash
pip install pandas openpyxl
```

### 数据准备

请将你的原始时间序列数据按照如下目录结构放置：

```
original_data/
└── Shanghai_T2DM/
    ├── 2000_0_20201230.xlsx
    ├── 2001_0_20201230.xlsx
    └── ... (其他文件)
```

经过pre_T2DM.py成功处理的数据和汇总表存放在以下目录：

```
generated_data/
└── Shanghai_T2DM/
└── Shanghai_T2DM_Summary.xlsx
```

最后经过T2DM.py成功处理的数据和汇总表存放在以下目录：
```
output/
└── Shanghai_T2DM/
└── Shanghai_T2DM_Summary.xlsx
```

### 使用说明

#### 先运行pre_T2DM.py在运行T2DM.py

#### pre_T2DM.py

此脚本用于处理合并单元格并生成新的数据文件。请确保文件路径和名称正确。

该代码主要是将表中CSII - basal insulin (Novolin R, IU / H)列的单元格进行拆分，以方便后续的数据处理。


#### T2DM.py

此脚本用于将处理后的数据与汇总表数据进行合并，并生成新的CSV和Excel文件。

该代码将总表中的个人性别、年龄、身高、体重与个人的表中的时间、CGM、胰岛素进行数据合并



### 结果存储

- 处理后的Excel文件存储在 `output/T2DM/excel/` 目录下
- 处理后的CSV文件存储在 `output/T2DM/csv/` 目录下

### 注意事项

- 确保所有数据文件路径正确。
- 确保数据文件格式与脚本要求一致。
- 在执行脚本前，请仔细检查文件路径和文件名是否正确。
