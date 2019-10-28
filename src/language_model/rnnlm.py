import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import numpy as np

class RNNLM(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(RNNLM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        self.rnn = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=False,
            dropout=self.dropout,
            batch_first=True
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.embed_size)
        )
        self.init_hidden = nn.Parameter(torch.FloatTensor(1, hidden_size))
        xavier_uniform_(self.init_hidden.data)

    def load_pretrained_embeddings(self, path: str, fixed: bool = False):
        embedding = np.load(path)
        self.embedding.weight.data.copy_(torch.tensor(embedding))
        if fixed:
            self.embedding.weight.requires_grad = False

    def forward(self, input):
        input = self.embedding(input)
        input = F.dropout(input, p=self.dropout, training=self.training)
        batch_size, time_step, embed_size = input.size()
        init_hidden = self.init_hidden.expand(batch_size, self.hidden_size).unsqueeze(0).contiguous()
        # hidden, _ = self.rnn(input, init_hidden)
        hidden, _ = self.rnn(input)
        logit = self.output_projection(hidden).matmul(self.embedding.weight.t())
        return logit