import math
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.duet.utils.tools import EarlyStopping, adjust_learning_rate
from ts_benchmark.utils.data_processing import split_before
from typing import Type, Dict, Optional, Tuple
from torch import optim
import numpy as np
import pandas as pd
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
    get_time_mark
)
from ts_benchmark.baselines.duet.models.duet_model import DUETModel
from ...models.model_base import ModelBase, BatchMaker

DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 2048,
    "hidden_size": 256,
    "freq": "h",
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "period_len": 4,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "moving_avg": 25,
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.02,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "huber",
    "patience": 10,
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "CI": True,
    "top_k": 5,
    "down_sampling_layers": 3,
    "down_sampling_window": 2,
    "down_sampling_method": "avg",
    "decomp_method":"moving_avg",
    "use_norm":0

}


class TransformerConfig:
    def __init__(self, **kwargs):
        # 遍历DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS字典，将键值对赋值给self
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)#setattr用于自动设置对象的属性

        # 遍历kwargs字典，将键值对赋值给self
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.horizon


class DUET(ModelBase):
    def __init__(self, **kwargs):
        super(DUET, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.scaler = StandardScaler()
        self.seq_len = self.config.seq_len
        self.win_size = self.config.seq_len

    @property
    def model_name(self):
        return "DUET"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm"
        }

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name
#__repr__方法是一个特殊的方法，用于定义对象的字符串表示形式，它返回一个字符串，该字符串可以包含对象的类型和值的信息。
#便于用户在print对象时看到设置的对象的信息，从而快速了解该对象
    def multi_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
#根据数据调整了以下超参数：self.config.freq(频率)、self.config.enc_in(编码器长度)、
#self.config.dec_in(解码器长度)、self.config.c_out(输出长度)、self.config.label_len(标签长度)


        # 获取训练数据的频率
        freq = pd.infer_freq(train_data.index)#pd.infer_freq()函数用于推断时间序列数据的频率,频率是指时间序列数据中时间戳之间的间隔
        # 如果频率为空，则抛出异常
        if freq == None:
            raise ValueError("Irregular time intervals")
        # 如果频率不在指定的范围内，则将频率设置为秒
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        # 否则，将频率设置为指定的值
        else:
            self.config.freq = freq[0].lower()

        # 获取训练数据的列数
        column_num = train_data.shape[1]
        # 设置编码器的输入维度
        self.config.enc_in = column_num#column_num表示数据的列数
        # 设置解码器的输入维度
        self.config.dec_in = column_num#column_num表示数据的列数
        # 设置输出的维度
        self.config.c_out = column_num

        # 如果模型名称为MICN，则将标签长度设置为序列长度
        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        # 否则，将标签长度设置为序列长度的一半
        else:
            setattr(self.config, "label_len", self.config.seq_len // 2)
#setattr()函数用于设置对象的属性，第一个参数是对象，第二个参数是属性名，第三个参数是属性值
    def single_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        # 获取训练数据的频率
        freq = pd.infer_freq(train_data.index)
        # 如果频率为空，则抛出异常
        if freq == None:
            raise ValueError("Irregular time intervals")
        # 如果频率不在指定的范围内，则将频率设置为秒
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        # 否则，将频率设置为指定的值
        else:
            self.config.freq = freq[0].lower()#freq[0].lower()将频率转换为小写字母.freq[0]表示频率的第一个字符

        # 获取训练数据的列数
        column_num = train_data.shape[1]
        # 设置编码器的输入维度
        self.config.enc_in = column_num
        # 设置解码器的输入维度
        self.config.dec_in = column_num
        # 设置输出的维度
        self.config.c_out = column_num

        # 设置标签的长度
        setattr(self.config, "label_len", self.config.horizon)

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        #检测任务超参数的调整（根据数据调整编码器和解码器的输入输出维度属性）
        # 获取训练数据的频率。
        freq = pd.infer_freq(train_data.index)
        # 如果索引频率为空，则抛出异常
        if freq == None:
            raise ValueError("Irregular time intervals")
        # 如果索引频率不在指定的范围内，则将频率设置为秒
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        # 否则，将索引频率设置为指定的频率
        else:
            self.config.freq = freq[0].lower()

        # 获取训练数据的列数
        column_num = train_data.shape[1]
        # 将编码器的输入维度设置为列数
        self.config.enc_in = column_num
        # 将解码器的输入维度设置为列数
        self.config.dec_in = column_num
        # 将输出的维度设置为列数
        self.config.c_out = column_num
        # 将标签的长度设置为48
        self.config.label_len = 48

    def padding_data_for_forecast(self, test):
        #该函数进行数据填充，便于预测任务
        #数据最新时间点往后填充horizon+1个时间点(全填充0)

        # 获取预测数据的索引列
        time_column_data = test.index
        # 获取预测数据的列名
        data_colums = test.columns
        # 获取预测数据的最后一个时间点
        start = time_column_data[-1]
        # padding_zero = [0] * (self.config.horizon + 1)
        # 生成从最后一个时间点到未来self.config.horizon + 1个时间点的日期序列
        date = pd.date_range(
            start=start, periods=self.config.horizon + 1, freq=self.config.freq.upper()#freq.upper()将频率转换为大写字母
        )#np.date_range()函数用于生成一个日期序列
        # 创建一个空的DataFrame，列名为data_colums
        df = pd.DataFrame(columns=data_colums)

        # 将DataFrame的前self.config.horizon + 1行填充为0
        df.iloc[: self.config.horizon + 1, :] = 0

        # 将生成的日期序列添加到DataFrame中，并设置为索引
        df["date"] = date
        df = df.set_index("date")
        # 将DataFrame的第一行删除
        new_df = df.iloc[1:]
        # 将新的DataFrame添加到预测数据中
        test = pd.concat([test, new_df])
        # 返回新的测试数据
        return test

    def _padding_time_stamp_mark(
        self, time_stamps_list: np.ndarray, padding_len: int
    ) -> np.ndarray:
        """
        Padding time stamp mark for prediction.

        :param time_stamps_list: A batch of time stamps.
        :param padding_len: The len of time stamp need to be padded.
        :return: The padded time stamp mark.
        """
        padding_time_stamp = []
        # 遍历时间戳列表
        for time_stamps in time_stamps_list:
            # 获取最后一个时间戳
            start = time_stamps[-1]
            # 根据配置文件中的频率，生成padding_len+1个时间戳
            expand_time_stamp = pd.date_range(
                start=start,
                periods=padding_len + 1,
                freq=self.config.freq.upper(),
            )
            #np.date_range()函数用于生成一个日期序列
            # 将生成的时间戳添加到padding_time_stamp列表中
            padding_time_stamp.append(expand_time_stamp.to_numpy()[-padding_len:])
        # 将padding_time_stamp列表转换为numpy数组
        padding_time_stamp = np.stack(padding_time_stamp)
        # 将原始时间戳列表和padding_time_stamp列表进行拼接
        whole_time_stamp = np.concatenate(
            (time_stamps_list, padding_time_stamp), axis=1
        )#np.concatenate()函数用于将两个数组沿着指定的轴进行拼接
        # 获取拼接后的时间戳标记
        padding_mark = get_time_mark(whole_time_stamp, 1, self.config.freq)
        return padding_mark
    def validate(self, valid_data_loader, criterion):
        #验证模型性能：在验证集上通过计算模型在验证集上的平均损失评估模型性能
        # 获取配置信息
        config = self.config
        total_loss = []
        self.model.eval()#将模型设置为评估模式，关闭dropout和batch normalization等训练时的操作
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#选择设备
        with torch.no_grad():#禁止使用梯度计算
            for input, target, input_mark, target_mark in valid_data_loader:#遍历验证数据加载器
                # 将输入数据和目标数据移动到设备上
                # input: [batch_size, seq_len, n_vars]
                # target: [batch_size, label_len + horizon, n_vars]
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )

                output, _ = self.model(input)#调用模型前向传播得到输出

                # 只取最后config.horizon个时间步的数据
                target = target[:, -config.horizon:, :]
                output = output[:, -config.horizon:, :]
                # 计算损失
                loss = criterion(output, target).detach().cpu().numpy()
                #criterion()函数用于计算损失,比较模型输出和目标值之间的差异，并将其作为损失值返回
                #detach()函数用于将张量从计算图中分离出来，返回一个新的张量，该张量与原始张量共享相同的数据，但不需要梯度计算
                #.cpu()函数用于将张量从GPU移动到CPU
                #numpy()函数用于将张量转换为numpy数组
            
                total_loss.append(loss)  # 将损失值添加到总损失列表中

        # 计算平均损失
        total_loss = np.mean(total_loss)
        self.model.train()#将模型设置为训练模式，开启dropout和batch normalization等训练时的操作
        return total_loss

    def forecast_fit(self, train_valid_data: pd.DataFrame, train_ratio_in_tv: float) -> "ModelBase":
        #输入训练数据和训练集验证集划分比例
        #输出训练好的模型对象，此处用“ModelBase”这个类名表示返回的对象是ModelBase类的实例
        """
        Train the model.

        :param train_data: Time data data used for training.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """
        #train_ratio_in_tv:表示训练集和验证集的划分比例，如果等于1，则表示不划分验证集
        # 如果训练数据只有一列，则进行单变量预测的超参数调优
        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        # 否则进行多变量预测的超参数调优
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)

        # 初始化模型
        self.model = DUETModel(self.config)

        print(
            "----------------------------------------------------------",
            self.model_name,
        )
        config = self.config
        # 将训练数据和验证数据分开
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        )

        # 对训练数据进行归一化
        self.scaler.fit(train_data.values)#self.scaler.fit()函数用于计算训练数据的均值和标准差
        # 将训练数据转换为数据集和数据加载器，为了方便加载和处理数据。数据加载器只能从数据集这个容器中才能读取
        # 如果训练集和验证集的划分比例不等于1，则将训练数据进行归一化
        # train_ratio_in_tv != 1:表示训练集和验证集的划分比例不等于1
        # train_data:训练数据，valid_data:验证数据

        if config.norm:
            train_data = pd.DataFrame(
                self.scaler.transform(train_data.values),#train_data.values:训练数据的值,为什么专门用values来取值，原因是因为train_data是一个DataFrame对象，values属性返回一个numpy数组，包含DataFrame中的所有数据，如果不用value取值的话,会报错
                #self.scaler.transform()函数用于将训练数据转换为归一化后的标准化数据，标准化数据是指将数据转换为均值为0，标准差为1的数据
                columns=train_data.columns,#将训练数据的列名赋值给train_data的列名
                index=train_data.index,
            )

        # 如果验证集不为空，则对验证数据进行归一化
        if train_ratio_in_tv != 1:
            if config.norm:
                valid_data = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index,
                )
            # 将验证数据转换为数据集和数据加载器
            #数据集：有多个数据结合的集合，数据加载器：用于从数据集中批量加载数据，防止一下子加载过多数据导致内存不足
            valid_dataset, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
            )

        # 将训练数据转换为数据集和数据加载器，原因是为了方便进行批量训练
        train_dataset, train_data_loader = forecasting_data_provider(#调用forecasting_data_provider()函数根据配置参数将训练数据转换为数据集和数据加载器
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,#是否打乱数据
            drop_last=train_drop_last,#是否丢掉最后一个不完整的批次
        )
    #数据-数据集-数据加载器，转化为数据加载器后可以高效完成对数据的一些加载处理，比如预处理、分批、打乱、设置多线程加载等，并提供统一的接口，不同类型的数据都可以通过数据加载器载入，更高效
        
        
        
        # Define the loss function and optimizer
        if config.loss == "MSE":
            criterion = nn.MSELoss()
        elif config.loss == "MAE":#如果config.loss等于"MAE"，则使用MAE损失函数
            criterion = nn.L1Loss()#nn.L1Loss()函数用于计算L1损失，即预测值和目标值之间的绝对误差之和
        else:
            criterion = nn.HuberLoss(delta=0.5)#nn.HuberLoss()函数用于计算Huber损失，即在L1损失和L2损失之间进行平滑过渡的损失函数

        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)#定义优化器，使用Adam优化算法，学习率为config.lr
    #优化器，完成模型参数的更新，根据梯度大小，按照参数=参数-梯度×学习率。学习率是一个控制模型参数更新步长的超参数。
    #不同的优化器有不同的更新方式，比如Adam、SGD、RMSprop等，不同的优化器有不同的特点，比如Adam可以自适应学习率，SGD可以设置学习率衰减等。
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = EarlyStopping(patience=config.patience)#该函数用于提前停止训练，当验证集损失不再下降时，提前停止训练，patience表示耐心值，具体指的是在验证集损失不再下降的情况下，等待多少个epoch后停止训练

        # 将模型和优化器移动到设备上
        #device:表示使用的设备，如果有GPU则使用GPU，否则使用CPU
        #self.model:模型对象，self.model.to(device):将模型移动到指定的设备上
        #self.model.to(device):将模型移动到指定的设备上

        #关于转移到device上的说明：
        #device:表示使用的设备，如果有GPU则使用GPU，否则使用CPU
        #为什么要转移到device上：因为GPU的计算速度比CPU快很多，所以在训练模型时，通常会将模型和数据转移到GPU上进行训练
        #需要转移到device上的对象有：1.模型（所有参数）2.任何参与计算的张量，比如输入数据。超参数配置不参与计算不用。两者必须在同一设备上否则报错




        self.model.to(device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total trainable parameters: {total_params}")

        for epoch in range(config.num_epochs):
            self.model.train()#将模型设置为训练模式，开启dropout和batch normalization等训练时的操作
            #这就是把模型赋给属性的原因，简化调用，可以无需重复创建模型类的实例，在多个方法中直接使用，并且可以保证模型参数及时更新。
            # for input, target, input_mark, target_mark in train_data_loader:
            for i, (input, target, input_mark, target_mark) in enumerate(
                    train_data_loader#enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个元组：（索引，值）
            ):
                optimizer.zero_grad()#本行代码用于将优化器的梯度清零，防止梯度累加
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )#多个对象进行同一个类型的操作就可以合并为一起a,b,c,d=(a.xx,b.xx,c.xx,d.xx)

                # decoder input

                output, loss_importance = self.model(input)#输入数据进入模型，开始训练，输出结果和重要性损失权重
                #output的shape是[batch_size, horizon, n_vars]，loss_importance的shape是[batch_size, seq_len, n_vars]
                target = target[:, -config.horizon:, :]#本行代码用于将目标数据的最后config.horizon个时间步的数据提取出来
                output = output[:, -config.horizon:, :]#本行代码用于将模型输出的最后config.horizon个时间步的数据提取出来
                loss = criterion(output, target)#本行代码用于计算模型输出和目标值之间的损失

                total_loss = loss + loss_importance#总损失等于模型输出和目标值之间的损失加上重要性损失
                #权重损失＋损失值也是一种加权方法，辅助加权
                #重要性损失指的是模型在训练过程中对不同时间步的预测结果的重要性进行加权，loss_importance是一个张量，表示每个时间步的损失值
                total_loss.backward()#本行代码用于计算梯度，即损失函数相对于模型参数的导数
                optimizer.step()#本行代码用于更新模型参数，即根据计算出的梯度和学习率来调整模型参数的值
            
            #验证集
            if train_ratio_in_tv != 1:
                valid_loss = self.validate(valid_data_loader, criterion)#输入验证集数据以及损失函数，返回验证集损失
                self.early_stopping(valid_loss, self.model)#输入验证集损失和模型的原因是：early_stopping类中有一个check_point属性，用于保存模型参数，当验证集损失不再下降时，保存模型参数，以便在训练过程中随时加载
                if self.early_stopping.early_stop:
                    break
                #防止过拟合的方法：早停机制。当验证集损失不再下降时，提前停止训练，防止模型过拟合。
                #原因：每一轮epoch,模型都会更新参数，如果模型在训练集上表现越来越好，但在验证集上表现越来越差，说明模型过拟合了，此时应该停止训练，防止模型过拟合。
                #在许多轮epoch之间，模型参数会继承，所以当验证集损失不再下降时，模型仍在更新参数导致拟合程度越来越高。
            adjust_learning_rate(optimizer, epoch + 1, config)#本行代码用于调整学习率，即根据训练的epoch数来调整学习率的大小，防止学习率过大导致模型训练不稳定

    def forecast(self, horizon: int, train: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        :param horizon: The predicted length.
        :param testdata: Time data data used for prediction.
        :return: An array of predicted results.
        """
        # 如果early_stopping.check_point不为None，则加载该checkpoint
        if self.early_stopping.check_point is not None:
            self.model.load_state_dict(self.early_stopping.check_point)#check_point是一个字典，包含了模型的参数和优化器的状态
        #如果触发了早停机制，则加载保存的模型参数
        # 如果config.norm为True，则对train进行归一化处理
        if self.config.norm:
            train = pd.DataFrame(
                self.scaler.transform(train.values),
                columns=train.columns,
                index=train.index,
            )

        # 如果model为None，则抛出ValueError异常
        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        # 获取config
        config = self.config
        # 将train和test进行分割
        train, test = split_before(train, len(train) - config.seq_len)

        # Additional timestamp marks required to generate transformer class methods
        test = self.padding_data_for_forecast(test)

        test_data_set, test_data_loader = forecasting_data_provider(
            test, config, timeenc=1, batch_size=1, shuffle=False, drop_last=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()#将模型设置为评估模式，关闭dropout和batch normalization等训练时的操作
        #dropout是指在训练过程中随机丢弃一部分神经元，以防止过拟合，batch normalization是指对每个batch的数据进行标准化处理，以加快模型收敛速度
        with torch.no_grad():#with torch.no_grad()用于禁止梯度计算，节省内存和计算资源
            #预测模式下要手动设置模型为eval模式，关闭一些操作，禁用梯度计算
            answer = None
            while answer is None or answer.shape[0] < horizon:
                for input, target, input_mark, target_mark in test_data_loader:
                    input, target, input_mark, target_mark = (
                        input.to(device),
                        target.to(device),
                        input_mark.to(device),
                        target_mark.to(device),
                    )

                    output, _ = self.model(input)

                column_num = output.shape[-1]
                temp = output.cpu().numpy().reshape(-1, column_num)[-config.horizon:]

                if answer is None:
                    answer = temp
                else:
                    answer = np.concatenate([answer, temp], axis=0)

                if answer.shape[0] >= horizon:
                    if self.config.norm:
                        answer[-horizon:] = self.scaler.inverse_transform(
                            answer[-horizon:]
                        )
                    return answer[-horizon:]

                output = output.cpu().numpy()[:, -config.horizon:, :]
                for i in range(config.horizon):
                    test.iloc[i + config.seq_len] = output[0, i, :]

                test = test.iloc[config.horizon:]
                test = self.padding_data_for_forecast(test)

                test_data_set, test_data_loader = forecasting_data_provider(
                    test,
                    config,
                    timeenc=1,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )

    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        """
        Make predictions by batch.

        :param horizon: The length of each prediction.
        :param batch_maker: Make batch data used for prediction.
        :return: An array of predicted results.
        """
        if self.early_stopping.check_point is not None:
            self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = input_data["input"]

        if self.config.norm:
            origin_shape = input_np.shape
            flattened_data = input_np.reshape((-1, input_np.shape[-1]))
            input_np = self.scaler.transform(flattened_data).reshape(origin_shape)

        input_index = input_data["time_stamps"]
        padding_len = (
            math.ceil(horizon / self.config.horizon) + 1
        ) * self.config.horizon
        all_mark = self._padding_time_stamp_mark(input_index, padding_len)

        answers = self._perform_rolling_predictions(horizon, input_np, all_mark, device)

        if self.config.norm:
            flattened_data = answers.reshape((-1, answers.shape[-1]))
            answers = self.scaler.inverse_transform(flattened_data).reshape(
                answers.shape
            )

        return answers

    def _perform_rolling_predictions(
        self,
        horizon: int,
        input_np: np.ndarray,
        all_mark: np.ndarray,
        device: torch.device,
    ) -> list:
        """
        Perform rolling predictions using the given input data and marks.

        :param horizon: Length of predictions to be made.
        :param input_np: Numpy array of input data.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param device: Device to run the model on.
        :return: List of predicted results for each prediction batch.
        """
        rolling_time = 0
        input_np, target_np, input_mark_np, target_mark_np = self._get_rolling_data(
            input_np, None, all_mark, rolling_time
        )
        with torch.no_grad():
            answers = []
            while not answers or sum(a.shape[1] for a in answers) < horizon:
                input, dec_input, input_mark, target_mark = (
                    torch.tensor(input_np, dtype=torch.float32).to(device),
                    torch.tensor(target_np, dtype=torch.float32).to(device),
                    torch.tensor(input_mark_np, dtype=torch.float32).to(device),
                    torch.tensor(target_mark_np, dtype=torch.float32).to(device),
                )
                output, _ = self.model(input)
                column_num = output.shape[-1]
                real_batch_size = output.shape[0]
                answer = (
                    output.cpu()
                    .numpy()
                    .reshape(real_batch_size, -1, column_num)[
                        :, -self.config.horizon :, :
                    ]
                )
                answers.append(answer)
                if sum(a.shape[1] for a in answers) >= horizon:
                    break
                rolling_time += 1
                output = output.cpu().numpy()[:, -self.config.horizon :, :]
                (
                    input_np,
                    target_np,
                    input_mark_np,
                    target_mark_np,
                ) = self._get_rolling_data(input_np, output, all_mark, rolling_time)

        answers = np.concatenate(answers, axis=1)
        return answers[:, -horizon:, :]

    def _get_rolling_data(
        self,
        input_np: np.ndarray,
        output: Optional[np.ndarray],
        all_mark: np.ndarray,
        rolling_time: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare rolling data based on the current rolling time.

        :param input_np: Current input data.
        :param output: Output from the model prediction.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param rolling_time: Current rolling time step.
        :return: Updated input data, target data, input marks, and target marks for rolling prediction.
        """
        if rolling_time > 0:
            input_np = np.concatenate((input_np, output), axis=1)
            input_np = input_np[:, -self.config.seq_len :, :]
        target_np = np.zeros(
            (
                input_np.shape[0],
                self.config.label_len + self.config.horizon,
                input_np.shape[2],
            )
        )
        target_np[:, : self.config.label_len, :] = input_np[
            :, -self.config.label_len :, :
        ]
        advance_len = rolling_time * self.config.horizon
        input_mark_np = all_mark[:, advance_len : self.config.seq_len + advance_len, :]
        start = self.config.seq_len - self.config.label_len + advance_len
        end = self.config.seq_len + self.config.horizon + advance_len
        target_mark_np = all_mark[
            :,
            start:end,
            :,
        ]
        return input_np, target_np, input_mark_np, target_mark_np
