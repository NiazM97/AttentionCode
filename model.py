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

        # We want the tensor to look like (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Now we register the tensor in the buffer of the model, so the tensor won't be changed by backprop
        self.register('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1]]).requires_grad(False)
        # False because we dont want the positions updated in training, they remain the same
        return self.dropout(x)

# ENCODER BLOCK

# Layer normalisation
# With a batch of n items, LN calculates the mean and variance for each iteam independently
# Then the new values fo reach are calculated using these
# gamma and beta are terms used by the model to amplify the values where needed

class LayerNormalisation(nn.Module):

    def __init__(self, eps: float= 10**-6) -> None:
        super().__init__()
        self.eps = eps # This is Epsilon in the equation and is used to avoid a situation where you divide by 0
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha + (x - mean) / (std + self.eps) + self.bias


# Feed forward
# In section 3.4, this is defined as two linear layers with a ReLU in between
# The first layer expands the dimension of the embeddings by a factor of 4 = d_ff
# The second layer reduces the dimension to return to d_model

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        #linear 2 applied on a normalised linear 1
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# Multi Headed Attention

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model : int, d_ff : int, heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        assert d_model % heads == 0, "d_model is not divisible by the number of heads"

        self.d_k = d_model // heads # same thing as d_v
        # the matrices that you multiply K, Q and V by, and the output matrix W_o
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    # used when you don't need to make an instance of the class
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # remember that masking is used to replace attention scores with 0 without affecting normalisation
        # we use the mask in the decoder block but not in the encoder block
        d_k = query.shape[-1]

        # Goes from (Batch, heads, seq_len, d_k) to (Batch, heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1))/ math.sqrt(d_k)
        # here the @ sign means matrix multiplication

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim= -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        # each of the above end up like: (Batch, seq_len, d_model)

        # Now we split the embedding into heads parts
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)
        # Now each look like (Batch, seq_len, heads, d_k)
        # After transposing, they look like: (Batch, heads, seq_len, d_k)
        # This is important because we want each head to see the whole sentence but only a part of the embedding

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask, self.dropout)

        # Now transpose to go from (Batch, heads, seq_len, d_k) to (Batch, seq_len, heads, d_k)
        # Then we transform to go to (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)

        return self.w_o(x)
    
# Residual (skip) connection

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        # This is the connection that adds the input to the output of the sublayer
        # It is used to prevent vanishing gradients and to allow the model to learn better
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalisation()

    def forward(self, x, sublayer):
        # sublayer is the output of the sublayer, e.g. attention or feed forward
        # x is the input to the sublayer
        return x + self.dropout(sublayer(self.norm(x)))



class EncoderBlock(nn.Module):

    # This is the block that contains the self attention and feed forward blocks
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
  
    def forward(self, x, src_mask):
        # x is the input to the encoder block, shape (Batch, seq_len, d_model)
        # src_mask is the mask for the source sequence, shape (Batch, 1, seq_len, seq_len)
        # The mask is used to prevent attention to padding tokens
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
# ENCODER
class Encoder(nn.Module):
    # This is the encoder that contains multiple encoder blocks
    # It also contains the input embeddings and positional encoding
    
    def __init__(self, layers: nn.ModuleList) -> None:
        # layers is a list of EncoderBlock instances
        # Each EncoderBlock contains a MultiHeadAttentionBlock and a FeedForwardBlock
        # The number of layers is the number of encoder blocks in the model
        # The input embeddings and positional encoding are applied before the encoder blocks
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
# DECODER BLOCK
class DecoderBlock(nn.Module):
    # This is the block that contains the self attention, encoder attention and feed forward blocks
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # src_mask is the mask for the source sequence, shape (Batch, 1, seq_len, seq_len) e.g. the source language
        # tgt_mask is the mask for the target sequence, shape (Batch, 1, seq_len, seq_len) e.g. the target language
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
# DECODER
class Decoder(nn.Module):
    # This is the decoder that contains multiple decoder blocks
    # It also contains the input embeddings and positional encoding
    def __init__(self, layers: nn.ModuleList) -> None:
        # layers is a list of DecoderBlock instances
        # Each DecoderBlock contains a MultiHeadAttentionBlock, a FeedForwardBlock and a CrossAttentionBlock
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

# Projection Layer

class ProjectionLayer(nn.Module):
    """ This is the layer that projects the output of the decoder to the vocabulary size.
        It is the linear layer after the decoder as well as the softmax layer"""
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
        # We use log softmax to get the log probabilities of the vocabulary words

# Transformer Model
# This is the main model that combines the encoder and decoder
class Transformer(nn.Module):
    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings, 
                 src_pos: PositionalEncoding, 
                 tgt_pos : PositionalEncoding, 
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # apply source embedding to source sentence
        src = self.src_embed(src)
        # apply positional encoding to source sentence
        src = self.src_pos(src)
        # encode
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # apply target embedding to target sentence
        tgt = self.tgt_embed(tgt)
        # apply positional encoding to target sentence
        tgt = self.tgt_pos(tgt)
        # decode
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # takes us from the embedding dimension to the vocabulary size
        return self.projection_layer(x)
    

# Build the Transformer model
# Given all the hyperparameters, we can build the model
def build_transformer(src_vocab_size: int, # vocabulary size of the source language
                      tgt_vocab_size: int, # vocabulary size of the target language
                      src_seq_len: int, # maximum length of the source sequence
                      tgt_seq_len: int, # maximum length of the target sequence
                      d_model: int = 512, # dimension of the model
                      N: int = 6, # number of encoder/decoder blocks
                      heads: int = 8, # number of attention heads
                      dropout: float = 0.1, # dropout rate
                      d_ff: int = 2048 # dimension of the feed forward layer
                      ) -> Transformer:
    # create embedding layers for source and target languages
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create positional encoding layers for source and target languages
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create N encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout) 
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # create N decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the weights of the model, using xavier initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

