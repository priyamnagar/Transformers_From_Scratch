# Python Libraries
import torch
import torch.nn as nn
from modules.multihead_attention import MultiHeadAttention
from modules.add_norm import AddNorm
from modules.ffn import FFN

class Encoder(nn.Module):
    def __init__(self,
                 batch_size, 
                 max_sentence_length,
                 embedding_dim,
                 num_multiheads,
                 num_encoders,
                 hidden_layer_size
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.max_sentence_length = max_sentence_length
        self.embedding_dim = embedding_dim
        self.input = None 
        self.num_multiheads = num_multiheads
        self.num_encoders = num_encoders
        self.hidden_layer_size = hidden_layer_size
        
        self.mask = torch.tril(torch.ones(self.max_sentence_length,self.max_sentence_length))
        self.mask[self.mask == 0] = -torch.inf
        self.mask[self.mask == 1] = 0
        
        # self.mask = None
        self.attention = MultiHeadAttention(self.embedding_dim, self.num_multiheads)
        self.add_norm = AddNorm(self.embedding_dim)
        self.ffn = FFN(self.embedding_dim, self.hidden_layer_size)
        
    def forward(self, input):
        self.input = input
        for i in range(self.num_encoders):
            
            contextual_embeddings, query, key, value = self.attention.multihead_attention(self.input, self.num_multiheads)
            contextual_embeddings_norm = self.add_norm.forward(self.input, contextual_embeddings)
            contextual_embeddings = self.ffn.forward(contextual_embeddings_norm)
            
            self.input = contextual_embeddings
            
        return contextual_embeddings, query, key, value
      