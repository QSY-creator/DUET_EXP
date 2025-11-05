from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
import torch.nn as nn
from einops import rearrange
from ts_benchmark.baselines.duet.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer
import torch

class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, num_nodes):
        super(GraphLearner, self).__init__()
        # 节点嵌入，用于学习每个变量的静态特性
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, hidden_size))
        # 两个线性层用于将时序特征映射到图学习空间
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        # x shape: [batch_size, seq_len, num_nodes]
        # 我们取最后一个时间步的特征作为代表，或使用全局池化
        x_last = x[:, -1, :] # [batch_size, num_nodes]

        # 计算自适应的邻接矩阵
        q = self.linear1(x_last).unsqueeze(1) # [batch_size, 1, num_nodes]
        k = self.linear2(x_last).unsqueeze(2) # [batch_size, num_nodes, 1]
        
        # 动态图结构：通过节点间特征交互学习
        dynamic_adj = F.softmax(F.leaky_relu(q + k), dim=-1) # [batch_size, num_nodes, num_nodes]
        
        # 静态图结构：从可学习的节点嵌入中学习
        static_adj = F.softmax(F.leaky_relu(self.node_embedding @ self.node_embedding.T), dim=-1)
        
        # 融合动态与静态图
        adj = dynamic_adj + static_adj
        
        return adj
class DUETModel(nn.Module):
    def __init__(self, config):
        super(DUETModel, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.graph_learner = GraphLearner(config.d_model, hidden_size=64, num_nodes=self.n_vars)
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
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

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        if self.n_vars > 1:
        # 1. 学习动态图关系
        # 我们需要将 temporal_feature reshape 以适应 graph_learner
        # temporal_feature shape is [B, d_model, n_vars], rearrange to [B, n_vars, d_model]
            node_features = temporal_feature.permute(0, 2, 1)
        # 用一个线性层将d_model降维到seq_len，或者直接用temporal_feature
        # 假设我们用一个线性层来获得节点表征
        # node_repr = self.repr_linear(temporal_feature)
        
        # 为了简化，我们直接从 temporal_feature 生成图
        # B x n_vars x d_model -> 我们需要池化d_model维度
            pooled_features = torch.mean(temporal_feature, dim=2) # -> B x n_vars
        # 这里需要一个更复杂的GraphLearner，我们先假设一个简化版
        # 假设 adj 的shape为 [B, n_vars, n_vars]
            adj_matrix = self.graph_learner(input.permute(0, 2, 1)) # GraphLearner输入需要 [B, N, L]

        # 2. 将图结构作为注意力偏置注入 Transformer
        # 将邻接矩阵 adj_matrix 传递给 Channel_transformer
            channel_group_feature, attention = self.Channel_transformer(x=temporal_feature, attn_mask=adj_matrix)
        
            output = self.linear_head(channel_group_feature)
        else:
            output = temporal_feature
            output = self.linear_head(output)

        output = rearrange(output, 'b n d -> b d n')
        output = self.cluster.revin(output, "denorm")
        return output, L_importance
