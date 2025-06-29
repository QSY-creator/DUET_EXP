import torch
import torch.nn as nn
from ..layers.Autoformer_EncDec import series_decomp



class Hybrid_extractor(nn.Module):
    def __init__(self, configs, individual=False):
        super(Hybrid_extractor, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.d_model
        self.decompsition = series_decomp(configs.moving_avg)
        
        # --- 保留原有的线性路径 ---
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # --- 新增：轻量级非线性路径 ---
        # 使用一个简单的MLP作为非线性特征提取器
        # 隐藏层维度可以设得小一些，以控制复杂度，例如 seq_len // 2
        hidden_dim = self.seq_len // 2
        self.NonLinear_Path = nn.Sequential(
            nn.Linear(self.seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.pred_len)
        )

        # --- 新增：自适应门控机制 ---
        # 这个门控决定了非线性路径的权重，它会学习输入序列的特征来做出判断
        self.gate = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),
            nn.Sigmoid() # Sigmoid输出在(0, 1)之间，非常适合做门控权重
        )
    def forecast(self, x_enc):
        # 1. 趋势和季节性分解 (与原来一致)
        seasonal_init, trend_init = self.decompsition(x_enc)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        # 2. 计算线性路径的输出 (与原来一致)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        linear_output = seasonal_output + trend_output

        # 3. 计算非线性路径的输出 (新增)
        # 我们让非线性路径直接作用在原始输入上（去掉分解），以捕捉更复杂的全局模式
        nonlinear_input = x_enc.permute(0, 2, 1) 
        nonlinear_output = self.NonLinear_Path(nonlinear_input)

        # 4. 计算门控权重 (新增)
        gate_weight = self.gate(nonlinear_input) # gate也从原始输入学习

        # 5. 融合线性和非线性输出 (核心)
        # 使用门控权重来动态融合
        # 当gate_weight接近1时，模型更依赖非线性路径
        # 当gate_weight接近0时，模型更依赖线性路径
        x = gate_weight * nonlinear_output + (1 - gate_weight) * linear_output
        
        return x.permute(0, 2, 1)

    def forward(self, x_enc):
        if x_enc.shape[0] == 0:
            return torch.empty((0, self.pred_len, 1 if configs.CI else configs.enc_in)).to(x_enc.device)
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]
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

