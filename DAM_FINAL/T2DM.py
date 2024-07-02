import argparse
import time
import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
import os
import argparse
import datetime
from datetime import datetime, timedelta
import glob
import time
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 随机数种子
np.random.seed(0)


class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


def plot_loss_data(data):
    # 使用Matplotlib绘制线图
    plt.figure()

    plt.plot(data, marker="o")

    # 添加标题
    plt.title("loss results Plot")

    # 显示图例
    plt.legend(["Loss"])

    plt.show()


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


def create_inout_sequences(input_data, tw, pre_len, config):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i : i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        if config.feature == "MS":
            train_label = input_data[:, -1:][i + tw : i + tw + pre_len]
        else:
            train_label = input_data[i + tw : i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def create_dataloader(config, device):

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    # print(config)
    dataframes = []
    # config.data_path = "data/T2DM/csv/2000_0_20201230out_put.csv"

    for index in range(2000, 2100):  # 这个循环会执行多次
        file_pattern = os.path.join("data/T2DM/csv/", f"{index}*.csv")
        all_files = glob.glob(file_pattern)

        # 打印文件名
        for file in all_files:
            filename = os.path.basename(file)
            # 读取 CSV 文件并添加到 dataframes 列表中
            df_single = pd.read_csv(file)
            dataframes.append(df_single)

    # 将所有 DataFrame 合并为一个
    df = pd.concat(dataframes, ignore_index=True)

    # print(df.shape)
    # print(df.columns)

    # 删除原始时间戳列
    pre_len = config.pre_len  # 预测未来数据的长度
    train_window = config.window_size  # 观测窗口

    # 编码时间信息为数值特征
    df[["Hour", "Minute"]] = df["Time"].str.split(":", expand=True)
    df.drop(columns=["Time"], inplace=True)
    df["Hour"] = pd.to_numeric(df["Hour"])
    df["Minute"] = pd.to_numeric(df["Minute"])

    # 保留特征列
    feature_columns = [
        "Hour",
        "Minute",
        "Gender (Female=1, Male=2)",
        "Age (years)",
        "Height (m)",
        "Weight (kg)",
        "CSII - basal insulin (Novolin R, IU / H)",
    ]
    # 将特征列和目标列分开
    feature_data = df[feature_columns].values
    target_data = df[[config.target]].values
    full_data = np.hstack((feature_data, target_data))

    scaler = StandardScaler()
    scaler.fit(full_data)
    full_data_normalized = scaler.transform(full_data)
    # 划分数据集
    train_data = full_data_normalized[int(0.3 * len(full_data)) :]
    valid_data = full_data_normalized[
        int(0.15 * len(full_data)) : int(0.30 * len(full_data))
    ]
    test_data = full_data_normalized[: int(0.15 * len(full_data))]
    print(
        "训练集尺寸:",
        len(train_data),
        "测试集尺寸:",
        len(test_data),
        "验证集尺寸:",
        len(valid_data),
    )

    # 转化为深度学习模型需要的类型Tensor
    train_data_normalized = torch.FloatTensor(train_data).to(device)
    test_data_normalized = torch.FloatTensor(test_data).to(device)
    valid_data_normalized = torch.FloatTensor(valid_data).to(device)

    # print("转化为深度学习模型需要的类型Tensor",train_data_normalized)

    # 定义训练器的输入
    train_inout_seq = create_inout_sequences(
        train_data_normalized, train_window, pre_len, config
    )
    test_inout_seq = create_inout_sequences(
        test_data_normalized, train_window, pre_len, config
    )
    valid_inout_seq = create_inout_sequences(
        valid_data_normalized, train_window, pre_len, config
    )

    # 创建数据集
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    valid_dataset = TimeSeriesDataset(valid_inout_seq)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    print(
        "通过滑动窗口共有训练集数据：",
        len(train_inout_seq),
        "转化为批次数据:",
        len(train_loader),
    )
    print(
        "通过滑动窗口共有测试集数据：",
        len(test_inout_seq),
        "转化为批次数据:",
        len(test_loader),
    )
    print(
        "通过滑动窗口共有验证集数据：",
        len(valid_inout_seq),
        "转化为批次数据:",
        len(valid_loader),
    )
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return train_loader, test_loader, valid_loader, scaler


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        rnn_num_layers=1,
        input_feature_len=1,
        sequence_len=168,
        hidden_size=100,
        bidirectional=False,
    ):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, input_seq):

        ht = torch.zeros(
            self.num_layers * self.rnn_directions,
            input_seq.size(0),
            self.hidden_size,
            device="cuda",
        )
        ct = ht.clone()
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        lstm_out, (ht, ct) = self.lstm(input_seq, (ht, ct))
        if self.rnn_directions > 1:
            lstm_out = lstm_out.view(
                input_seq.size(0),
                self.sequence_len,
                self.rnn_directions,
                self.hidden_size,
            )
            lstm_out = torch.sum(lstm_out, axis=2)
        return lstm_out, ht.squeeze(0)


class AttentionDecoderCell(nn.Module):
    def __init__(self, input_feature_len, out_put, sequence_len, hidden_size):
        super().__init__()
        # attention - inputs - (decoder_inputs, prev_hidden)
        self.attention_linear = nn.Linear(hidden_size + input_feature_len, sequence_len)
        # attention_combine - inputs - (decoder_inputs, attention * encoder_outputs)
        self.decoder_rnn_cell = nn.LSTMCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, input_feature_len)

    def forward(self, encoder_output, prev_hidden, y):
        if prev_hidden.ndimension() == 3:
            prev_hidden = prev_hidden[-1]  # 保留最后一层的信息
        attention_input = torch.cat((prev_hidden, y), axis=1)
        attention_weights = F.softmax(
            self.attention_linear(attention_input), dim=-1
        ).unsqueeze(1)
        attention_combine = torch.bmm(attention_weights, encoder_output).squeeze(1)
        rnn_hidden, rnn_hidden = self.decoder_rnn_cell(
            attention_combine, (prev_hidden, prev_hidden)
        )
        output = self.out(rnn_hidden)
        return output, rnn_hidden


class EncoderDecoderWrapper(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        pred_len,
        window_size,
        teacher_forcing=0.3,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(num_layers, input_size, window_size, hidden_size)
        self.decoder_cell = AttentionDecoderCell(
            input_size, output_size, window_size, hidden_size
        )
        self.output_size = output_size
        self.input_size = input_size
        self.pred_len = pred_len
        self.teacher_forcing = teacher_forcing
        self.linear = nn.Linear(input_size, output_size)

    def __call__(self, xb, yb=None):
        input_seq = xb
        encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden
        if torch.cuda.is_available():
            outputs = torch.zeros(
                self.pred_len, input_seq.size(0), self.input_size, device="cuda"
            )
        else:
            outputs = torch.zeros(input_seq.size(0), self.output_size)
        y_prev = input_seq[:, -1, :]
        for i in range(self.pred_len):
            if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                y_prev = yb[:, i].unsqueeze(1)
            rnn_output, prev_hidden = self.decoder_cell(
                encoder_output, prev_hidden, y_prev
            )
            y_prev = rnn_output
            outputs[i, :, :] = rnn_output
        outputs = outputs.permute(1, 0, 2)
        if self.output_size == 1:
            outputs = self.linear(outputs)
        return outputs


def train(model, args, scaler, device):
    start_time = time.time()  # 计算起始时间
    model = model
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs = args.epochs
    model.train()  # 训练模式
    results_loss = []
    for i in tqdm(range(epochs)):
        losss = []
        for seq, labels in train_loader:
            optimizer.zero_grad()

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)

            single_loss.backward()

            optimizer.step()
            losss.append(single_loss.detach().cpu().numpy())
        tqdm.write(f"\t Epoch {i + 1} / {epochs}, Loss: {sum(losss) / len(losss)}")
        results_loss.append(sum(losss) / len(losss))

        torch.save(model.state_dict(), "T2DM_final_model/save_model.pth")
        time.sleep(0.1)

    # valid_loss = valid(model, args, scaler, valid_loader)
    # 尚未引入学习率计划后期补上
    # 保存模型

    print(
        f">>>>>>>>>>>>>>>>>>>>>>模型已保存,用时:{(time.time() - start_time) / 60:.4f} min<<<<<<<<<<<<<<<<<<"
    )
    plot_loss_data(results_loss)


def valid(model, args, scaler, valid_loader):
    lstm_model = model
    # 加载模型进行预测
    lstm_model.load_state_dict(torch.load("T2DM_final_model/save_model.pth"))
    lstm_model.eval()  # 评估模式
    losss = []

    for seq, labels in valid_loader:
        pred = lstm_model(seq)
        mae = calculate_mae(
            pred.detach().numpy().cpu(), np.array(labels.detach().cpu())
        )  # MAE误差计算绝对值(预测值  - 真实值)
        losss.append(mae)

    print("验证集误差MAE:", losss)
    return sum(losss) / len(losss)


def test(model, args, test_loader, scaler):
    # 加载模型进行预测
    losss = []
    model = model
    model.load_state_dict(torch.load("T2DM_final_model/save_model.pth"))
    model.eval()  # 评估模式
    results = []
    labels = []
    for seq, label in test_loader:
        pred = model(seq)
        mae = calculate_mae(
            pred.detach().cpu().numpy(), np.array(label.detach().cpu())
        )  # MAE误差计算绝对值(预测值  - 真实值)
        losss.append(mae)
        pred = pred[:, 0, :]
        label = label[:, 0, :]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i][-1])
            labels.append(label[i][-1])
    plt.figure(figsize=(10, 5))
    print("测试集误差MAE:", losss)
    print("测试集整体的MAE:", mean_absolute_error(results, labels))
    print("测试集整体的MSE:", mean_squared_error(results, labels))
    # 绘制历史数据
    plt.plot(labels, label="TrueValue")

    # 绘制预测数据
    # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
    plt.plot(results, label="Prediction")

    # 添加标题和图例
    plt.title("test state")
    plt.legend()
    plt.show()


# 检验模型拟合情况
def inspect_model_fit(model, args, valid_loader, scaler):
    # Load the saved model state
    model.load_state_dict(torch.load("T2DM_final_model/save_model.pth"))
    model.eval()  # Set model to evaluation mode
    results = []
    labels = []

    # Iterate over the validation data
    for seq, label in valid_loader:
        # Move tensors to the correct device
        seq, label = seq.to(device), label.to(device)

        with torch.no_grad():
            pred = model(seq)

        pred = pred[:, 0, :]
        label = label[:, 0, :]

        # Inverse transform the predictions and labels
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i][-1])
            labels.append(label[i][-1])

    # Compute the overall MAE
    overall_mae = mean_absolute_error(labels, results)
    print("验证集整体的MAE:", overall_mae)
    print("验证集整体的MSE:", mean_squared_error(results, labels))

    # Plot the true values and predictions
    plt.plot(labels, label="True Value")
    plt.plot(results, label="Prediction")
    plt.title("Validation Set - True Values vs Predictions")
    plt.legend()
    plt.show()


def predict(model, args, device, scaler, train_loader):
    # 存储所有样本的预测结果和标签
    results = []
    labels = []

    # 获取训练集中的所有样本
    all_samples = list(train_loader)
    num_samples = len(all_samples)

    # 迭代多个样本
    for sample_idx in range(num_samples):
        sample_seq, sample_label = all_samples[sample_idx]

        # 将样本数据转换到指定设备
        sample_seq, sample_label = sample_seq.to(device), sample_label.to(device)

        # 模型预测
        model.load_state_dict(torch.load("T2DM_final_model/save_model.pth"))
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            pred = model(sample_seq)

        pred = pred[:, 0, :]  # 只选择第一个时间步的预测结果
        sample_label = sample_label[:, 0, :]  # 只选择第一个时间步的标签

        # 反向转换预测结果和原始数据
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        sample_label = scaler.inverse_transform(sample_label.detach().cpu().numpy())

        # 将每个样本的预测结果和标签添加到结果列表中
        results.append(pred.squeeze())  # squeeze() 可以去除形状中的单维度条目
        labels.append(sample_label.squeeze())

    # 将结果和标签转换为 numpy 数组
    results = np.array(results)
    labels = np.array(labels)

    # 计算每个样本的 MAE
    mae_list = np.mean(np.abs(results - labels), axis=1)

    # 计算所有样本的平均 MAE
    # avg_mae = np.mean(mae_list)

    overall_mae = mean_absolute_error(labels, results)
    print(f"Average MAE across all samples: {overall_mae:.2f}")

    # 绘制 MAE 分布图
    plt.figure(figsize=(12, 6))
    plt.hist(mae_list, bins=20, edgecolor="black")
    plt.title("MAE Distribution Across All Samples")
    plt.xlabel("MAE")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series forecast")
    parser.add_argument("-model", type=str, default="LSTM", help="模型持续更新")
    parser.add_argument(
        "-window_size",
        type=int,
        default=128,
        help="时间窗口大小, window_size > pre_len",
    )
    parser.add_argument("-pre_len", type=int, default=4, help="预测未来数据长度")
    # data
    parser.add_argument(
        "-shuffle",
        action="store_true",
        default=True,
        help="是否打乱数据加载器中的数据顺序",
    )
    parser.add_argument("-data_path", type=str, default="none", help="你的数据数据地址")
    parser.add_argument(
        "-target",
        type=str,
        default="CGM (mg / dl)",
        help="你需要预测的特征列，这个值会最后保存在csv文件里",
    )
    parser.add_argument("-input_size", type=int, default=8, help="特征个数")
    parser.add_argument("-output_size", type=int, default=1, help="输出特征数")
    parser.add_argument(
        "-feature",
        type=str,
        default="MS",
        help="[M, S, MS],多元预测多元,单元预测单元,多元预测单元",
    )

    # learning
    parser.add_argument("-lr", type=float, default=0.0001, help="学习率")
    parser.add_argument(
        "-drop_out", type=float, default=0.02, help="随机丢弃概率,防止过拟合"
    )
    parser.add_argument("-epochs", type=int, default=40, help="训练轮次")
    parser.add_argument("-batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("-save_path", type=str, default="models")

    # model
    parser.add_argument("-hidden_size", type=int, default=128, help="隐藏层单元数")
    parser.add_argument("-laryer_num", type=int, default=2)

    # device
    parser.add_argument("-use_gpu", type=bool, default=True)
    parser.add_argument(
        "-device", type=int, default=0, help="只设置最多支持单个gpu训练"
    )

    # option
    parser.add_argument("-train", type=bool, default=False)
    parser.add_argument("-test", type=bool, default=True)
    parser.add_argument("-inspect_fit", type=bool, default=True)
    parser.add_argument("-predict", type=bool, default=True)
    args = parser.parse_args()

    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f"{args.device}")
    else:
        device = torch.device("cpu")
    print("使用设备:", device)
    train_loader, test_loader, valid_loader, scaler = create_dataloader(args, device)

    if args.feature == "MS" or args.feature == "S":
        args.output_size = 1
    else:
        args.output_size = args.input_size

    # 实例化模型
    try:
        print(
            f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )
        model = EncoderDecoderWrapper(
            args.input_size,
            args.output_size,
            args.hidden_size,
            args.laryer_num,
            args.pre_len,
            args.window_size,
        ).to(device)
        print(
            f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )
    except:
        print(
            f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型失败<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )

    # 训练模型
    if args.train:
        print(
            f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )
        train(model, args, scaler, device)
    if args.test:
        print(
            f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )
        test(model, args, test_loader, scaler)
    if args.inspect_fit:
        print(
            f">>>>>>>>>>>>>>>>>>>>>>>>>开始检验{args.model}模型拟合情况<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )
        inspect_model_fit(model, args, valid_loader, scaler)
    if args.predict:
        print(
            f">>>>>>>>>>>>>>>>>>>>>>>>>开始检验{args.model}预测开始<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )
        predict(model, args, device, scaler, train_loader)
    plt.show()
