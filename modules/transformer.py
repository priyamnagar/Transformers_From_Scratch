# Our Modules
from modules.input import Input
from modules.encoder import Encoder
from modules.decoder import Decoder
import torch.nn as nn
import torch.nn.functional as F
import torch

class Transformer(nn.Module):
    def __init__(self,
                batch_size,
                max_sentence_length,
                embedding_dim,
                num_multiheads,
                num_encoders,
                num_decoders,
                encoder_hidden_layer_size,
                decoder_hidden_layer_size,
                english_vocab_size,
                french_vocab_size
                ):
        
        super().__init__()

        self.batch_size = batch_size
        self.max_sentence_length = max_sentence_length
        self.embedding_dim = embedding_dim
        self.num_multiheads = num_multiheads
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.encoder_hidden_layer_size = encoder_hidden_layer_size
        self.decoder_hidden_layer_size = decoder_hidden_layer_size
        self.english_vocab_size = english_vocab_size
        self.french_vocab_size = french_vocab_size

        self.inp_english = Input(
            self.max_sentence_length,
            vocab_size = self.english_vocab_size,
            embedding_dim = self.embedding_dim
        )

        self.inp_french = Input(
            self.max_sentence_length,
            vocab_size = self.french_vocab_size,
            embedding_dim = self.embedding_dim
        )

        self.encoder = Encoder(
            self.batch_size,
            self.max_sentence_length,
            self.embedding_dim,
            self.num_multiheads,
            self.num_encoders,
            self.encoder_hidden_layer_size
        )

        self.decoder = Decoder(
            self.max_sentence_length,
            self.embedding_dim,
            self.num_multiheads,
            self.num_decoders,
            self.decoder_hidden_layer_size
        )

        self.linear = nn.Linear(self.embedding_dim, self.french_vocab_size)

    def forward(self, english_batch, french_batch):
        device = english_batch.device
        # Encoder
        encoder_input = self.inp_english.forward(english_batch).to(device)

        encoder_output = self.encoder.forward(encoder_input).to(device)
    
        # Decoder
        decoder_input = self.inp_french.forward(french_batch).to(device)
    
        decoder_output = self.decoder.forward(decoder_input, encoder_output).to(device)

        probabilities = F.softmax(self.linear(decoder_output)).to(device)
        # result = torch.argmax(probabilities, dim=-1)

        return probabilities