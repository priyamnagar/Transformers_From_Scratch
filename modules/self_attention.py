import torch
import torch.nn as nn

class SelfAttention():
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.softmax = nn.Softmax(dim = 2)

    def self_attention(self, query, key, value, mask = None):
        k_t = key.transpose(1, 2)
        scores = torch.bmm(query, k_t) / (self.input_dim ** 0.5)
        if(mask != None):
            scores = scores + mask
        attention = self.softmax(scores)
        contextual_emb = torch.bmm( attention, value)
        return contextual_emb