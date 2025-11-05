# ts_benchmark/baselines/duet/models/duet_model.py

from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
import torch.nn as nn
from einops import rearrange
from ts_benchmark.baselines.duet.utils.masked_attention import Encoder, EncoderLayer, AttentionLayer, FullAttention
# +++ 导入我们新的模块 +++
from .clustered_attention import ClusteredAttentionLayer 
import torch

class DUETModel(nn.Module):
    def __init__(self, config):
        super(DUETModel, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in

        # --- 模块化、工程化的改造 ---
        if self.n_vars > 1:
            # 定义超参数
            num_clusters = 4 # 可调
            e_layers = config.e_layers if hasattr(config, 'e_layers') else 2
            
            # 我们将 Channel_transformer 的层替换为我们的新模块
            # 这里我们只替换第一层，以引入聚类机制，后续层可以继续进行标准的自注意力处理
            layers = []
            
            # 第一层：使用我们的聚类注意力模块
            layers.append(
                ClusteredAttentionLayer(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    num_clusters=num_clusters,
                    dropout=config.dropout,
                    activation=config.activation
                )
            )

            # 后续层 (如果 e_layers > 1): 使用标准的自注意力 EncoderLayer
            # 这允许模型在聚类信息融合后，进一步进行全局的特征提炼
            for _ in range(e_layers - 1):
                layers.append(
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(mask_flag=False), # No mask needed for channel self-attention
                            config.d_model,
                            config.n_heads
                        ),
                        config.d_model,
                        config.d_ff,
                        dropout=config.dropout,
                        activation=config.activation
                    )
                )

            self.Channel_transformer = nn.ModuleList(layers)
            self.norm_layer = nn.LayerNorm(config.d_model)

        self.linear_head = nn.Sequential(nn.Linear(config.d_model, config.pred_len), nn.Dropout(config.fc_dropout))

    def forward(self, input):
        # x: [batch_size, seq_len, n_vars]
        if self.CI:
            channel_independent_input = rearrange(input, 'b l n -> (b n) l 1')
            reshaped_output, L_importance = self.cluster(channel_independent_input)
            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input.shape[0])
        else:
            temporal_feature, L_importance = self.cluster(input)

        # temporal_feature shape: [B, d_model, n_vars]
        # permute to [B, n_vars, d_model] for channel-wise processing
        channel_feature = temporal_feature.permute(0, 2, 1)

        if self.n_vars > 1:
            # 将特征输入到我们改造后的 Channel Transformer
            attns = []
            for layer in self.Channel_transformer:
                channel_feature, attn = layer(channel_feature)
                attns.append(attn)
            
            channel_feature = self.norm_layer(channel_feature)
            output = self.linear_head(channel_feature)
        else:
            # 单变量情况
            output = self.linear_head(channel_feature)

        output = rearrange(output, 'b n d -> b d n')
        output = self.cluster.revin(output, "denorm")
        return output, L_importance