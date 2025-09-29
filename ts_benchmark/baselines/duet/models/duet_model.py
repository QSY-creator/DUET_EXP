from ts_benchmark.baselines.duet.layers.linear_extractor_cluster import Linear_extractor_cluster
import torch.nn as nn
from einops import rearrange
from ts_benchmark.baselines.duet.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer
import torch
from TimePro import Model as TimeProModel

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
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )
        self.TPM=TimeProModel(config)
        self.linear_head = nn.Sequential(nn.Linear(config.d_model, config.pred_len), nn.Dropout(config.fc_dropout))

    def forward(self, input):
        # x: [batch_size, seq_len, n_vars](b,l,n)
        
        if self.CI:
            channel_independent_input = rearrange(input, 'b l n -> (b n) l 1')

            reshaped_output, L_importance = self.TPM(channel_independent_input)

            temporal_feature = rearrange(reshaped_output, '(b n) l 1 -> b l n', b=input.shape[0])

        else:
            temporal_feature= self.TPM(input)

        # B x d_model x n_vars -> B x n_vars x d_model
        
        #在timepro内进行了修改，使其自己进行了内部投影，把bne->bnd,之后的结果就是b n d,不用转换
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
