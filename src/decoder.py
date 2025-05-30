import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = EncoderBlock(embed_size, num_heads, forward_expansion, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.norm(attention + x)
        out = self.transformer_block(query, enc_out, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, forward_expansion, dropout, max_length):
        super().__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, num_heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length)
        
        x = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions)
        )
        
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)
            
        return self.fc_out(x)