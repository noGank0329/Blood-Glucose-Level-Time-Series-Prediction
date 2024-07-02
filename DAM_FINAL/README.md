## README

### 项目概述

本项目实现了一个用于时间序列预测的模型，主要使用了LSTM（长短期记忆网络）和注意力机制。该模型能够对时间序列数据进行训练、验证和测试，并提供预测结果的可视化。

### 环境配置

在开始使用本项目之前，请确保你的环境已经安装了以下依赖包：

- Python 3.6+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

你可以通过以下命令安装这些依赖：

```bash
pip install torch numpy pandas matplotlib scikit-learn tqdm
```

### 数据准备

请将你的时间序列数据按照如下格式放置于 `data/T2DM/csv/` 目录下：

```
data/
├── T2DM/
│   └── csv/
│       ├── 2000_0_20201230.csv
│       ├── 2001_0_20201230.csv
│       └── ... (其他文件)
|
|── T1DM/
│   └── csv/
│       ├── 1001_0_20210730.csv
│       └── ... (其他文件)
```

数据文件应为 CSV 格式，并包含以下列：

- Time: 时间戳
- Gender (Female=1, Male=2): 性别
- Age (years): 年龄
- Height (m): 身高
- Weight (kg): 体重
- CSII - basal insulin (Novolin R, IU / H): 基础胰岛素
- CGM (mg / dl): 连续血糖监测数据（预测目标列）

### 使用说明

#### 参数说明

在运行脚本时，你可以通过命令行参数来配置模型和数据加载器。以下是一些常用参数：

- `-model`: 模型名称，默认为"LSTM"
- `-window_size`: 时间窗口大小，默认为128
- `-pre_len`: 预测未来数据长度，默认为4
- `-shuffle`: 是否打乱数据加载器中的数据顺序，默认是
- `-data_path`: 数据文件路径，默认为"none"
- `-target`: 需要预测的特征列，默认为"CGM (mg / dl)"
- `-input_size`: 特征个数，默认为8
- `-output_size`: 输出特征数，默认为1
- `-feature`: 特征类型，默认为"MS"（多元预测单元）
- `-lr`: 学习率，默认为0.0001
- `-drop_out`: 随机丢弃概率，默认为0.02
- `-epochs`: 训练轮次，默认为40
- `-batch_size`: 批次大小，默认为64
- `-save_path`: 模型保存路径，默认为"models"
- `-hidden_size`: 隐藏层单元数，默认为128
- `-laryer_num`: LSTM层数，默认为2
- `-use_gpu`: 是否使用GPU，默认为True
- `-device`: GPU设备编号，默认为0
- `-train`: 是否训练模型，默认为False
- `-test`: 是否测试模型，默认为True
- `-inspect_fit`: 是否检验模型拟合情况，默认为True
- `-predict`: 是否进行预测，默认为True

#### 运行训练

```bash
python script.py -train True -data_path "data/T2DM/csv/" -epochs 50
```

#### 运行测试

```bash
python script.py -test True -data_path "data/T2DM/csv/"
```

#### 检验模型拟合情况

```bash
python script.py -inspect_fit True -data_path "data/T2DM/csv/"
```

#### 运行预测

```bash
python script.py -predict True -data_path "data/T2DM/csv/"
```

### 代码结构

- `StandardScaler`: 用于数据标准化的类
- `plot_loss_data(data)`: 用于绘制损失图的函数
- `TimeSeriesDataset`: 自定义数据集类
- `create_inout_sequences`: 用于创建时间序列数据的函数
- `calculate_mae`: 计算平均绝对误差的函数
- `create_dataloader`: 创建数据加载器的函数
- `LSTMEncoder`: LSTM编码器类
- `AttentionDecoderCell`: 带注意力机制的解码器单元类
- `EncoderDecoderWrapper`: 编码器-解码器包装类
- `train`: 训练模型的函数
- `valid`: 验证模型的函数
- `test`: 测试模型的函数
- `inspect_model_fit`: 检验模型拟合情况的函数
- `predict`: 进行预测的函数

### 结果可视化

- `plot_loss_data(data)`: 用于绘制训练过程中的损失变化
- `test`: 在测试集上进行预测并绘制真实值与预测值的对比图
- `inspect_model_fit`: 在验证集上进行预测并绘制真实值与预测值的对比图
- `predict`: 绘制预测误差的分布图

### 注意事项

- 数据集应具有一致的时间戳格式
- 确保数据文件路径正确
- 根据硬件条件选择是否使用GPU进行训练

### 参考

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [Matplotlib 官方文档](https://matplotlib.org/stable/contents.html)
- [scikit-learn 官方文档](https://scikit-learn.org/stable/documentation.html)

