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
    
    
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers_num, dropout):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layers_num, batch_first=True, bidirectional=True)
        self.init_state = nn.Parameter(torch.zeros(2*2*layers_num, 1, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, lengths, **kwargs):
        self.lstm.flatten_parameters()
        init_state = self.init_state.repeat([1, inputs.size(0), 1])
        cell_init, hidden_init = init_state[:init_state.size(0)//2], init_state[init_state.size(0)//2:]
        idx = torch.argsort(lengths, descending=True)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs[idx], lengths[idx], batch_first=True)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            self.lstm(packed_inputs, (hidden_init, cell_init))[0], batch_first=True)
        return self.dropout(outputs[torch.argsort(idx)])
    
    

class LstmTAE(nn.Module):
    def __init__(self,labels_num, emb_size, hidden_size, layers_num, linear_size, dropout, 
                 vocab_size=None, emb_init=None, emb_trainable=True, padding_idx=0, emb_dropout=0.2, **kwargs):
        super(LstmTAE, self).__init__()
        self.emb = Embedding(vocab_size, emb_size, emb_init, emb_trainable, padding_idx, emb_dropout)
        self.lstm = LSTMEncoder(emb_size, hidden_size, layers_num, dropout)
        self.tae = TAE(30,512,hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, labels_num)
        self.drop_out = nn.Dropout(0.2)
        nn.init.xavier_uniform_(self.fc1.weight)


    def forward(self, inputs, **kwargs):
        emb_out, lengths, masks = self.emb(inputs, **kwargs)
        rnn_out = self.lstm(emb_out, lengths.cpu())   
        top_out = self.tae(rnn_out,masks)
        top_out = self.drop_out(top_out)
        logits = self.fc1(top_out)

        return logits