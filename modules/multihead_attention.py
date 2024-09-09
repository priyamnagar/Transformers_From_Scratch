import torch.nn as nn
from modules.self_attention import SelfAttention
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 embedding_dim,
                 num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.transform = nn.Linear(self.embedding_dim, self.embedding_dim)

    def attention(self, x, mask, query, key, value):
        heads = []
        for i in range(self.num_heads):
            sa = SelfAttention(self.embedding_dim // self.num_heads)
            contextual_embeddings = sa.self_attention( query, key, value, mask)
            heads.append(contextual_embeddings)
        cat_emb = torch.cat(heads, dim=2)
        contextual_embeddings = self.transform(cat_emb)
        return contextual_embeddings

    def multihead_attention(self, x):
        Q = nn.Linear(self.embedding_dim, self.embedding_dim//self.num_heads)
        K = nn.Linear(self.embedding_dim, self.embedding_dim//self.num_heads)
        V = nn.Linear(self.embedding_dim, self.embedding_dim//self.num_heads)
        query = Q(x)
        key = K(x)
        value = V(x)
        
        contextual_embeddings = self.attention(x, mask = None, query = query, key = key, value = value)
        return contextual_embeddings

    def masked_multihead_attention(self, x, mask):
        Q = nn.Linear(self.embedding_dim, self.embedding_dim//self.num_heads)
        K = nn.Linear(self.embedding_dim, self.embedding_dim//self.num_heads)
        V = nn.Linear(self.embedding_dim, self.embedding_dim//self.num_heads)
        query = Q(x)
        key = K(x)
        value = V(x)

        contextual_embeddings = self.attention(x, mask = mask, query = query, key = key, value = value)
        return contextual_embeddings

    def cross_multihead_attention(self, x, query, key, value):
        contextual_embeddings = self.attention(x, mask = None, query = query, key = key, value = value)
        return contextual_embeddings
        

    def forward(self, 
                x,
                type = "multihead",
                mask = None,
                query = None,
                key = None,
                value = None
                ):
        if(type == "multihead"):
            contextual_embeddings = self.multihead_attention(x)
            
        elif(type == "masked"):
            if(mask == None):
                raise "Mask not provided for masked multihead attention"
            else:
                contextual_embeddings = self.masked_multihead_attention(x, mask)

        elif(type == "cross"):
            if(query == None or key == None or value == None):
                raise "Required input not provided for cross attention"
            else:
                contextual_embeddings = self.cross_multihead_attention(x, query, key, value)

        else:
            raise "Wrong multihead attention type"

        return contextual_embeddings

        
        
        
