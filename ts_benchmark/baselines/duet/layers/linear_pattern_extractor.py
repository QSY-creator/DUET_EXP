import torch
import torch.nn as nn
from ..layers.Autoformer_EncDec import series_decomp
import math

class Hybrid_extractor(nn.Module):
    def __init__(self, configs, individual=False):
        super(Hybrid_extractor, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.d_model
        self.enc_in = 1 if configs.CI else configs.enc_in
        # 注意：这里我们不再需要 self.enc_in，因为我们将在 forecast 中处理
        self.decompsition = series_decomp(configs.moving_avg)
        
        # --- 线性路径 (保持不变) ---
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # --- 非线性路径 (输入保持一致) ---
        # 现在，非线性路径也作用于分解后的季节性部分，以学习非线性的季节性模式
        hidden_dim = self.seq_len // 4 # 可以适当减小隐藏层维度，进一步控制复杂度
        self.NonLinear_Seasonal = nn.Sequential(
            nn.Linear(self.seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.pred_len)
        )

        # --- 自适应门控机制 ---
        # 门控也基于更平滑的季节性输入来做决策
        self.gate = nn.Linear(self.seq_len, self.pred_len) # 先只用一个Linear层

        # ***** 关键修复 1: 门控初始化 *****
        # 我们将门控的偏置初始化为一个较大的负数。
        # 这样，在训练开始时，Sigmoid(gate_output)会接近0。
        # 这意味着模型在开始时几乎完全依赖于鲁棒的线性路径，避免了噪声干扰。
        with torch.no_grad():
            self.gate.bias.fill_(-5.0)

    def forecast(self, x_enc):
        # 1. 趋势和季节性分解
        seasonal_init, trend_init = self.decompsition(x_enc)
        seasonal_init_p = seasonal_init.permute(0, 2, 1)
        trend_init_p = trend_init.permute(0, 2, 1)
        
        # 2. 线性路径 (现在只作用于趋势)
        # 趋势通常是线性的，让线性层专门处理它
        trend_output = self.Linear_Trend(trend_init_p)

        # ***** 关键修复 2: 统一输入路径 *****
        # 3. 季节性路径 (混合线性和非线性)
        # 线性部分
        linear_seasonal_output = self.Linear_Seasonal(seasonal_init_p)
        # 非线性部分
        nonlinear_seasonal_output = self.NonLinear_Seasonal(seasonal_init_p)
        # 门控权重
        gate_weight = torch.sigmoid(self.gate(seasonal_init_p)) # Sigmoid在这里用

        # 融合季节性输出
        seasonal_output = gate_weight * nonlinear_seasonal_output + (1 - gate_weight) * linear_seasonal_output

        # 4. 最终融合
        # 将处理好的趋势和季节性部分相加
        x = seasonal_output + trend_output
        
        return x.permute(0, 2, 1)

    def forward(self, x_enc):
        # 你的forward函数基本没问题，但为了安全，我们用configs里的设定
        # 注意，我移除了__init__中的self.enc_in，因为configs里有
        if x_enc.shape[0] == 0:
            # 这里的configs需要从外部传入，或者假设它可以被访问
            # 在DUET的框架下，configs是可用的
            enc_in = self.enc_in
            return torch.empty((0, self.pred_len, enc_in)).to(x_enc.device)
        
        dec_out = self.forecast(x_enc)
        return dec_out # 原forward的最后一步切片是不必要的，因为输出维度已经是pred_len
class Linear_extractor(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Linear_extractor, self).__init__()
        self.seq_len = configs.seq_len

        self.pred_len = configs.d_model
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.enc_in = 1 if configs.CI else configs.enc_in
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))



    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)


    def forward(self, x_enc):
        if x_enc.shape[0] == 0:
            return torch.empty((0, self.pred_len, self.enc_in)).to(x_enc.device)
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

