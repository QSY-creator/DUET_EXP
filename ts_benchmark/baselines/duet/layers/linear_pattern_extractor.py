import torch
import torch.nn as nn
from ..layers.Autoformer_EncDec import series_decomp
from ts_benchmark.baselines.duet.layers.layers import *
from ts_benchmark.baselines.duet.layers.patch_layer import *


class Linear_extractor(nn.Module):

    #每个专家的处理过程
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Linear_extractor, self).__init__()
        self.seq_len = configs.seq_len
        configs.CI=0
        self.pred_len = configs.d_model
        self.enc_in = 1 if configs.CI else configs.enc_in
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.n_cluster = 4
        self.d_ff = configs.d_ff
        self.n_vars = configs.enc_in
        
        self.device='cuda:0' #注意这里会和原本的device冲突,只不过在这个文件里因为individual被写死为c,所以不会遇到
        if self.individual==1 or self.individual=='True' or self.individual=='true':
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
        elif self.individual==0 or self.individual=='False' or self.individual=='false':
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        elif self.individual=='c':
            self.Linear_Seasonal = Cluster_wise_linear(self.n_cluster, self.channels, self.seq_len, self.pred_len, self.device)
            self.Linear_Trend = Cluster_wise_linear(self.n_cluster, self.channels,self.seq_len, self.pred_len, self.device)
        if self.individual == "c":
            self.Cluster_assigner = Cluster_assigner(self.channels, self.n_cluster, self.seq_len, self.d_ff, device=self.device)
            self.cluster_emb = self.Cluster_assigner.cluster_emb

    def encoder(self, x):
        if self.individual == "c":
            self.cluster_prob, cluster_emb = self.Cluster_assigner(x, self.cluster_emb)
            
        else:
            self.cluster_prob = None
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual==1 or self.individual=='True' or self.individual=='true':
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        elif self.individual==0 or self.individual=='False' or self.individual=='false':
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        elif self.individual == "c":
            seasonal_output = self.Linear_Seasonal(seasonal_init, self.cluster_prob)
            trend_output = self.Linear_Trend(trend_init, self.cluster_prob)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)


    def forward(self, x_enc):
        if x_enc.shape[0] == 0:
            return torch.empty((0, self.pred_len, self.enc_in)).to(x_enc.device)
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]#这里为什么要这么取，也有必要吗？感觉需要调整

