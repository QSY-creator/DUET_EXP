import torch
import torch.nn as nn
from ..layers.Autoformer_EncDec import series_decomp

from mamba_ssm import Mamba

class MambaExpert(nn.Module):
    """
    一个基于Mamba的状态空间模型专家。
    用于捕捉复杂的、动态的、非线性的模式。
    【说明】 这个专家现在接收一个config对象，而不是多个独立参数，以匹配你现有代码的风格。
    """
    def __init__(self, config):
        super(MambaExpert, self).__init__()
        # 从config中获取必要的参数
        seq_len = config.seq_len
        pred_len = config.pred_len
        d_model = getattr(config, 'd_model', 128) # 使用getattr提供默认值
        d_state = getattr(config, 'd_state', 16)
        d_conv = getattr(config, 'd_conv', 4)
        expand = getattr(config, 'expand', 2)

        # 输入投影层，将单变量序列映射到Mamba的维度
        # 【说明】 这里的输入维度是config.enc_in，因为你的数据可能是多变量的。
        # 如果是单变量，config.enc_in就是1。
        self.input_proj = nn.Linear(config.enc_in, d_model)
        
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # 输出投影层，将Mamba的输出映射到预测长度
        self.output_proj = nn.Linear(d_model * seq_len, pred_len * config.enc_in)
        self.pred_len = pred_len
        self.enc_in = config.enc_in

    def forward(self, x):
        # x shape: [expert_batch_size, seq_len, enc_in]
        # Mamba需要 (batch, seq_len, d_model) 的输入
        
        # 【修改】 调整以适应输入形状
        B, L, C = x.shape
        x_proj = self.input_proj(x) # [B, L, d_model]
        
        mamba_out = self.mamba(x_proj) # [B, L, d_model]
        
        # 将输出展平并通过线性层进行预测
        mamba_out_flat = mamba_out.flatten(start_dim=1) # [B, L * d_model]
        prediction = self.output_proj(mamba_out_flat) # [B, pred_len * enc_in]
        
        # 【修改】 调整输出形状以匹配 [expert_batch_size, pred_len, enc_in]
        prediction = prediction.view(B, self.pred_len, self.enc_in)
        
        return prediction


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

