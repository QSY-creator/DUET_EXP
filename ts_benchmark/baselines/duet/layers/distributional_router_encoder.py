import torch
import torch.nn as nn



# distributional_router_encoder.py (修改版)
class encoder(nn.Module):
    def __init__(self, config):
        super(encoder, self).__init__()
        # 假设 config.enc_in 是通道数，现在输入维度翻倍
        # 注意：你需要修改模型配置，让新的input_size能被正确传入
        input_size = config.enc_in * 2 # 均值+标准差
        num_experts = config.num_experts
        encoder_hidden_size = config.hidden_size

        # !!! 这里的input_size需要从外部配置传入正确的维度 !!!
        self.distribution_fit = nn.Sequential(
            nn.Linear(input_size, encoder_hidden_size, bias=False), 
            nn.ReLU(),
            nn.Linear(encoder_hidden_size, num_experts, bias=False)
        )

    def forward(self, x):
        # x 形状: [batch_size, seq_len, num_channels]
        # 我们在通道维度上计算统计量
        mean = torch.mean(x, dim=1) # -> [batch_size, num_channels]
        std = torch.std(x, dim=1)   # -> [batch_size, num_channels]
        
        # 将特征拼接
        # -> [batch_size, num_channels * 2]
        rich_features = torch.cat([mean, std], dim=-1) 
        
        out = self.distribution_fit(rich_features)
        return out


# class encoder(nn.Module):
#     def __init__(self, config):
#         super(encoder, self).__init__()
#         input_size = config.seq_len
#         num_experts = config.num_experts
#         encoder_hidden_size = config.hidden_size

#         self.distribution_fit = nn.Sequential(nn.Linear(input_size, encoder_hidden_size, bias=False), nn.ReLU(),
#                                               nn.Linear(encoder_hidden_size, num_experts, bias=False))

#     def forward(self, x):
#         mean = torch.mean(x, dim=-1)
#         out = self.distribution_fit(mean)
#         return out
