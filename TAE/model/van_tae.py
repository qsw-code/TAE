import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tae import TAE

class Embedding(nn.Module):
    def __init__(self, vocab_size=None, emb_size=None, emb_init=None, emb_trainable=True, padding_idx=0, dropout=0.2):
        super(Embedding, self).__init__()
        if emb_init is not None:
            if vocab_size is not None:
                assert vocab_size == emb_init.shape[0]
            if emb_size is not None:
                assert emb_size == emb_init.shape[1]
            vocab_size, emb_size = emb_init.shape
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx, sparse=True,
                                _weight=torch.from_numpy(emb_init).float() if emb_init is not None else None)
        self.emb.weight.requires_grad = emb_trainable
        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx

    def forward(self, inputs):
        emb_out = self.dropout(self.emb(inputs))
        lengths, masks = (inputs != self.padding_idx).sum(dim=-1), inputs != self.padding_idx
        return emb_out[:, :lengths.max()], lengths, masks[:, :lengths.max()]
    
    

class VANTAE(nn.Module):
    def __init__(self, labels_num, emb_size, 
                 vocab_size=None, emb_init=None, emb_trainable=True, padding_idx=0, emb_dropout=0.2, **kwargs):
        super(VANTAE, self).__init__()
        self.emb = Embedding(vocab_size, emb_size, emb_init, emb_trainable, padding_idx, emb_dropout)
        self.tae = TAE(30,emb_size,emb_size)
        self.fc1 = nn.Linear(emb_size, labels_num)
        self.drop_out = nn.Dropout(0.2)


    def forward(self, inputs, **kwargs):
        emb_out, lengths, masks = self.emb(inputs, **kwargs)
        top_out = self.tae(emb_out,masks)
        top_out = self.drop_out(top_out)
        logits = self.fc1(top_out)

        return logits