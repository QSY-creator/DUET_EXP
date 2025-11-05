# ts_benchmark/baselines/duet/models/clustered_attention.py

import torch
import torch.nn as nn
from ts_benchmark.baselines.duet.utils.masked_attention import AttentionLayer, FullAttention

class ClusteredAttentionLayer(nn.Module):
    """
    A self-contained module for Hierarchical Clustered Routing Attention.
    It replaces a standard self-attention block.
    """
    def __init__(self, d_model, n_heads, d_ff, num_clusters, dropout=0.1, activation="relu"):
        super(ClusteredAttentionLayer, self).__init__()
        self.d_model = d_model
        self.num_clusters = num_clusters

        # 1. Learnable Cluster Tokens
        self.cluster_tokens = nn.Parameter(torch.randn(1, self.num_clusters, d_model))

        # 2. Attention layers for the two stages
        # Stage 1: Variables -> Clusters (Cross-Attention)
        self.v2c_attention = AttentionLayer(FullAttention(mask_flag=False), d_model, n_heads)
        # Stage 2: Clusters -> Variables (Cross-Attention)
        self.c2v_attention = AttentionLayer(FullAttention(mask_flag=False), d_model, n_heads)

        # 3. Standard Transformer components (FFN, Norm, Dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        :param x: Input tensor of shape [Batch, Num_Vars, d_model]
        :return: Output tensor of the same shape
        """
        B = x.shape[0]
        # Expand cluster tokens for the batch
        batch_cluster_tokens = self.cluster_tokens.expand(B, -1, -1)

        # --- Stage 1: Variables aggregate information into Cluster Tokens ---
        # Q: cluster_tokens, K: x, V: x
        updated_clusters, _ = self.v2c_attention(batch_cluster_tokens, x, x, attn_mask=None)
        
        # Add & Norm for the cluster tokens
        updated_clusters = self.norm1(batch_cluster_tokens + self.dropout(updated_clusters))

        # --- Stage 2: Cluster Tokens distribute information back to Variables ---
        # Q: x, K: updated_clusters, V: updated_clusters
        infused_info, _ = self.c2v_attention(x, updated_clusters, updated_clusters, attn_mask=None)

        # The main residual connection is on `x`
        x = self.norm2(x + self.dropout(infused_info))
        
        # --- Final Feed-Forward Network ---
        y = self.ffn(x)
        x = x + self.dropout(y)
        
        # The original EncoderLayer returns attention, we can return None
        return x, None