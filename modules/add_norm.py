import torch.nn as nn
import torch

class AddNorm(nn.Module):
    def __init__(self,
                 embedding_dim,
                 eps = 1e-5
                ):
        super().__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
        self.gamma = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, input_vectors, contextual_embeddings):
        # Layer Normalization
        mean = contextual_embeddings.mean(dim = -1, keepdim = True)
        var = contextual_embeddings.var(dim = -1, keepdim = True, unbiased = False)

        contextual_embeddings_norm = (contextual_embeddings - mean) / torch.sqrt(var + self.eps)
        contextual_embeddings_norm = self.gamma * contextual_embeddings_norm + self.beta

        # Residual Connection
        contextual_embeddings_norm = contextual_embeddings_norm + input_vectors

        return contextual_embeddings_norm
        