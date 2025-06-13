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
        # x: [batch_size, seq_len, n_vars]
        if self.CI:
            channel_independent_input = rearrange(input, 'b l n -> (b n) l 1')
            ###方案一
            ###
            #这里加下采样,并把下采样结果变成channel_independent_input的形状,对于每一个长度的进行cluster                    
            #下面是添加进来的多尺度模块，开始修改

            #如果应用未来特征..，但是我们没有用到
            if self.use_future_temporal_feature:
                if self.channel_independence == 1:
                    B, T, N = x_enc.size()
                    x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                    self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
                else:
                    self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            #开始下采样
            x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)##下采样，输入时间序列BLN，输出[BLN;BLN;..]

            x_list = []
            x_mark_list = []
            if x_mark_enc is not None:
                for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                    B, T, N = x.size()
                    x = self.normalize_layers[i](x, 'norm')
                    if self.channel_independence == 1:
                        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                        x_mark = x_mark.repeat(N, 1, 1)
                    x_list.append(x)
                    x_mark_list.append(x_mark)
            else:
                for i, x in zip(range(len(x_enc)), x_enc, ):
                    B, T, N = x.size()
                    x = self.normalize_layers[i](x, 'norm')#归一化
                    if self.channel_independence == 1:         
                        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)
    ###

            #此处要将x_list作为输入输进去，检查x_list的格式以及channel_independent_input的格式
            #检查x_mark_enc的处理
            for i in x_list:



            ###
            #开始进行MOE
                reshaped_output, L_importance = self.cluster(channel_independent_input)
            ###
            #这里加尺度混合,整合为多尺度混合后的时序特征,形状要是(bn) l 1
            


            ###
        ###
        #方案二，在每个MOE里专家接受到任务后开始下采样，然后分别对每个进行提取，再进行混合输出半成品
        ###

        ###修改结束,        
            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input.shape[0])

        else:#这边是通道不独立的分支，未作多尺度的修改
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
