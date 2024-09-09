import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self,
                embedding_dim,
                hidden_layer_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_layer_size = hidden_layer_size
        
        self.hidden_layer = nn.Linear(in_features = self.embedding_dim , out_features = self.hidden_layer_size)
        self.output_layer = nn.Linear(in_features = self.hidden_layer_size, out_features = self.embedding_dim)

    def forward(self, x):
        h1 = F.relu(self.hidden_layer(x))
        out = self.output_layer(h1)
        return out