# Python Libraries
import torch
import torch.nn as nn
import numpy as np

class Input(nn.Module):
    def __init__(self,
                 max_sentence_length,
                 vocab_size,
                 embedding_dim
                ):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.max_sentence_length = max_sentence_length
        self.embedding_dim = embedding_dim

    def positional_encoding(self):
        encoding = np.zeros((self.max_sentence_length, self.embedding_dim))
        
        # Compute the position values (pos) for each dimension (2i and 2i+1)
        for pos in range(self.max_sentence_length):
            for i in range(0, self.embedding_dim, 2):
                encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i)/self.embedding_dim)))
                encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i)/self.embedding_dim)))
        return torch.Tensor(encoding)

    def forward(self, 
                batch):
        embeddings = self.embedding_layer(batch)
                
        pos_enc = self.positional_encoding()
        embeddings = embeddings + pos_enc
        
        return embeddings

    