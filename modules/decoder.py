# Python Libraries
import torch
import torch.nn as nn
from modules.multihead_attention import MultiHeadAttention
from modules.add_norm import AddNorm
from modules.ffn import FFN

class Decoder(nn.Module):
    def __init__(self,
                 max_sentence_length,
                 embedding_dim,
                 num_multiheads,
                 num_decoders,
                 hidden_layer_size
                ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_multiheads = num_multiheads
        self.max_sentence_length = max_sentence_length
        self.num_decoders = num_decoders
        self.hidden_layer_size = hidden_layer_size
        
        self.mask = torch.tril(torch.ones(self.max_sentence_length,self.max_sentence_length))
        self.mask[self.mask == 0] = -torch.inf
        self.mask[self.mask == 1] = 0
        
        self.attention = MultiHeadAttention(self.embedding_dim, self.num_multiheads)
        self.add_norm = AddNorm(self.embedding_dim)
        self.ffn = FFN(self.embedding_dim, self.hidden_layer_size)
        self.input = None

    def forward(self,input, encoder_output):
        self.input = input
        for i in range(self.num_decoders):
            # maksed multihead attention block
            contextual_embeddings = self.attention.forward(self.input, type = "masked", mask = self.mask)
            
            contextual_embeddings_norm = self.add_norm.forward(self.input, contextual_embeddings)
            
            # Cross multihead attention block
            Q = nn.Linear(self.embedding_dim, self.embedding_dim//self.num_multiheads)
            K = nn.Linear(self.embedding_dim, self.embedding_dim//self.num_multiheads)
            V = nn.Linear(self.embedding_dim, self.embedding_dim//self.num_multiheads)
            
            query = Q(contextual_embeddings_norm)
            key = K(encoder_output)
            value = V(encoder_output)
            
            contextual_embeddings = self.attention.forward(self.input, type = "cross", query = query, key = key, value = value )
            
            contextual_embeddings_norm = self.add_norm.forward(self.input, contextual_embeddings)

            # Feed forward block
            contextual_embeddings = self.ffn.forward(contextual_embeddings_norm)
            
            contextual_embeddings_norm = self.add_norm.forward(self.input, contextual_embeddings)

            
            self.input = contextual_embeddings
            
        return contextual_embeddings