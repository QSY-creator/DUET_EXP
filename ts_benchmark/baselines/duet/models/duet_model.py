from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
import torch.nn as nn
from einops import rearrange
from ts_benchmark.baselines.duet.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer
import torch


class DUETModel(nn.Module):
    def __init__(self, config):
        super(DUETModel, self).__init__()
        self.cluster = Linear_extractor_cluster(config)
        self.CI = config.CI
        self.n_vars = config.enc_in
        self.mask_generator = Mahalanobis_mask(config.seq_len)
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
                for _ in range(config.e_layers)   #这是一个列表推导式,只不过比较长，[]for i in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)#归一化层
        )

        self.linear_head = nn.Sequential(nn.Linear(config.d_model, config.pred_len), nn.Dropout(config.fc_dropout))

    def forward(self, input):
        #方案一：在每个moe之前完成下采样，并将每个尺度的都送进moe进行处理，但问题是每个moe内部就进行趋势分解和合并了，输出了完整的，没法进行趋势和季节分别尺度混合在混合。
        #我有一个大胆的想法，moe内部混合完之后，我外部再进行混合，然后再进行多尺度混合，但是我不是很确定，原本的是对谁进行季节和趋势的分解？也是完整的吗？是的，是完整的，直接输入归一化后的各种尺度组成的列表。然后直接输入到趋势分解函数中。
        #目前采用的是方案二，即在每个moe内部进行
        #如何实现跨分支的实验记录？
        #实际上timemixser后面还有一部分代码没看，不知道作用
        #启示：下一次看代码做思维导图的时候要详细，要把内容尽可能地完善，各种情况下的，不然我确实后面没有心力再去看了，并且的确有可能遇到
        #如何实验自己的更改是否有效，需要把代码提交上去再跑，还是暂存就可以了？另外我跑实验本地没法跑linux命令，想起来了好像可以。修改下命令行？真的吗，我怎么记得上一次在linux上也需要修改？
        # x: [batch_size, seq_len, n_vars]
        #发现个问题为什么切换分支之后，打开的代码还是原分支的修改没有回到该分支的版本？--没有commit!
        if self.CI:
            channel_independent_input = rearrange(input, 'b l n -> (b n) l 1')

            reshaped_output, L_importance = self.cluster(channel_independent_input)

            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input.shape[0])

        else:
            temporal_feature, L_importance = self.cluster(input)

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        if self.n_vars > 1:
            changed_input = rearrange(input, 'b l n -> b n l')
            channel_mask = self.mask_generator(changed_input)

            channel_group_feature, attention = self.Channel_transformer(x=temporal_feature, attn_mask=channel_mask)

            output = self.linear_head(channel_group_feature)
        else:
            output = temporal_feature
            output = self.linear_head(output)

        output = rearrange(output, 'b n d -> b d n')
        output = self.cluster.revin(output, "denorm")
        return output, L_importance
