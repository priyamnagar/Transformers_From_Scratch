# Transformers_From_Scratch
This is an attempt to create Transformers model from scratch.  We train a transformers model on machine translation dataset using Pytorch.

# Dataset
The Dataset contains an English to French translation data, you can find it in the /Data folder.

# How to Use
To Train or inference the transformer, you need to use the Transformer class present in /modules/transformer.py.

_transformer = Transformer(
    batch_size, # Data Batch size
    max_sentence_length, # Maximum sentence length allowed for input and output
    embedding_dim, # Dimention of word embeddings ex: 512
    num_multiheads, # Number of Self attention heads used in Multihead attention block
    num_encoders, # Number of encoder layers to use 
    num_decoders, # Number of decoder layers to use
    encoder_hidden_layer_size, # Hidden layer size for Fully connected neural network block in encoder
    decoder_hidden_layer_size, # Hidden layer size for Fully connected neural network block in decoder
    english_vocab_size = preprocess.english_vocab_size + 1, # English Vocabulary size , here we are using a preprocess class.
    french_vocab_size = preprocess.french_vocab_size + 1 # French Vocabulary size , here we are using a preprocess class.
)_

To get predictions you can use the forward method defined in transformers class as below:

_decoder_prediction = transformer.forward(encoder_input, decoder_input).float()_

here encoder_input is the english input to the decoder (dimension is (batch_size, max_sentence_length, embedding_dim))
and decoder_input is the french input to decoder with dimension same as the encoder_input.

