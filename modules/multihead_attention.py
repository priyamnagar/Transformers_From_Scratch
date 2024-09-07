import torch.nn as nn
from modules.self_attention import SelfAttention
import torch

class MultiHeadAttention():
    def __init__(self, input_dim, num_heads):
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.Q = nn.Linear(self.input_dim, self.input_dim//num_heads)
        self.K = nn.Linear(self.input_dim, self.input_dim//num_heads)
        self.V = nn.Linear(self.input_dim, self.input_dim//num_heads)
        self.transform = nn.Linear(self.input_dim, self.input_dim)
        
    def multihead_attention(self, x, num_heads, mask = None):
        query = self.Q(x)
        key = self.K(x)
        value = self.V(x)
        
        heads = []
        for i in range(num_heads):
            sa = SelfAttention(self.input_dim // num_heads)
            contextual_embeddings = sa.self_attention( query, key, value, mask)
            heads.append(contextual_embeddings)
        cat_emb = torch.cat(heads, dim=2)
        contextual_embeddings = self.transform(cat_emb)
        return contextual_embeddings
        