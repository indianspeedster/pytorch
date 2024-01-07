import torch
import torch.nn as nn
import math

class InputEmbedding(nn.module):
    def __init__(self, d_model: int, vocab_size:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size = vocab_size
        self.embedding = nn.embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


    

class PositionalEncoding(nn.module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)   

        # Create a matrix of [seq_len, d_model] where each row is a positional encoding    
        pe = torch.zeros(seq_len, d_model)
        #create a vector of shape(seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0) # [1, seq_len, d_model]

        # Register the buffer so it will be saved when calling torch.save

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)