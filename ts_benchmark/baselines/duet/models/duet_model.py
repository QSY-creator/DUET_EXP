# ts_benchmark/baselines/duet/models/duet_model.py

from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
import torch.nn as nn
from einops import rearrange
from ts_benchmark.baselines.duet.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer
import torch
import torch.nn.functional as F # <--- 已添加导入

# +++ START: 新增的、经过重构的 GraphLearner 模块 +++
class GraphLearner(nn.Module):
    """
     learns a dynamic and static adjacency matrix.
    """
    def __init__(self, num_nodes, embedding_dim, alpha=3):
        super(GraphLearner, self).__init__()
        self.num_nodes = num_nodes
        
        # 静态图嵌入：捕捉变量间的固有、静态关系
        self.static_embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        
        # 动态图学习的线性变换层
        self.linear_q = nn.Linear(embedding_dim, embedding_dim)
        self.linear_k = nn.Linear(embedding_dim, embedding_dim)
        
        self.alpha = alpha

    def forward(self, node_features):
        """
        :param node_features: [Batch, num_nodes, embedding_dim]
        :return: adj_matrix: [Batch, num_nodes, num_nodes]
        """
        # --- 学习动态图 ---
        # node_features 已经是每个通道经过第一阶段专家模型提取的特征
        q = self.linear_q(node_features) # [B, N, D]
        k = self.linear_k(node_features) # [B, N, D]
        
        # 使用点积注意力计算动态邻接矩阵
        dynamic_adj = torch.matmul(q, k.transpose(1, 2)) # [B, N, N]
        
        # --- 学习静态图 ---
        static_adj = F.relu(torch.matmul(self.static_embedding, self.static_embedding.transpose(0, 1)))
        
        # --- 融合图结构 ---
        # 使用 softmax 确保边的权重是归一化的
        # alpha 控制静态图的影响力
        adj_matrix = F.softmax(F.leaky_relu(dynamic_adj) + self.alpha * static_adj.unsqueeze(0), dim=-1)
        
        return adj_matrix
# +++ END: GraphLearner 模块 +++


class DUETModel(nn.Module):
    def __init__(self, config):
        super(DUETModel, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        
        # --- 替换 Mahalanobis_mask 为 GraphLearner ---
        if self.n_vars > 1:
            self.graph_learner = GraphLearner(num_nodes=self.n_vars, embedding_dim=config.d_model)

        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True, # mask_flag 现在用于指示是否使用注意力偏置
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )

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
        temporal_feature = temporal_feature.permute(0, 2, 1)

        if self.n_vars > 1:
            # 1. 使用 temporal_feature (每个通道的浓缩特征) 来学习图结构
            adj_matrix = self.graph_learner(temporal_feature)

            # 2. 将图结构作为注意力偏置注入 Transformer
            channel_group_feature, attention = self.Channel_transformer(x=temporal_feature, attn_mask=adj_matrix)
            output = self.linear_head(channel_group_feature)
        else:
            # 单变量情况下，无需图学习和通道Transformer
            output = self.linear_head(temporal_feature)

        output = rearrange(output, 'b n d -> b d n')
        output = self.cluster.revin(output, "denorm")
        return output, L_importance