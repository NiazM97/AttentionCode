# Making the transformer model from the "Attention is all you need" paper from scratch
# Refer to Figure 1 in the paper to see the model architecture

import torch
import torch.nn as nn
import math

from networkx.generators.degree_seq import directed_configuration_model


# Input embeddings
    # First the input gets an ID based on the words position in the vocabulary "table"
    # Then it gets a vector embedding

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        # d_model is the dimension of the model/ vector
        # vocab_size is the number of words in the vocabulary
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
        # In section 3.4 in the paper, we multiply the embedding by sqrt of the dimension of the model

# Positional Encoding
    # This adds another vector of the same dimension to the embedding which tells it about the positional
    # information of the word in the sentence

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
    # seq_len is the maximum length of the sentence, since we create a vector for each position
    # dropout is a technique to make the model less prone to overfitting
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # we need seq_len number of vectors, each of which are d_model long: matrix.shape = (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # from equations in section 3.5 of the paper:
        # Create a vector of shape (seq_len,1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # denominator is in log space instead for some numerical stability
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))

        # Apply the sin of the equation to even position words
        pe[:,0::2] = torch.sin(position * denominator)
        # Apply the cos of the equation to odd position words
        pe[:,1::2] = torch.cos(position * denominator)

