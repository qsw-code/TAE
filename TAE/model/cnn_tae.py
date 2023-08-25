import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tae import TAE

class CNNTAE(nn.Module):
    def __init__(self, dropout, labels_num, dynamic_pool_length, bottleneck_dim, num_filters,
                 vocab_size=None, emb_size=None, emb_trainable=True, emb_init=None, padding_idx=0, **kwargs):
        super(CNNTAE, self).__init__()
        if emb_init is not None:
            if vocab_size is not None:
                assert vocab_size == emb_init.shape[0]
            if emb_size is not None:
                assert emb_size == emb_init.shape[1]
            vocab_size, emb_size = emb_init.shape            
        
            
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx, sparse=True,
                                _weight=torch.from_numpy(emb_init).float() if emb_init is not None else None)
        self.emb.weight.requires_grad = emb_trainable
        
        self.ks = 3 # There are three conv nets here
        ## Different filter sizes in xml_cnn than kim_cnn
        self.conv1 = nn.Conv2d(1, num_filters, (2, emb_size), padding=(1,0))
        self.conv2 = nn.Conv2d(1, num_filters, (4, emb_size), padding=(3,0))
        self.conv3 = nn.Conv2d(1, num_filters, (8, emb_size), padding=(7,0))

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters*self.ks, labels_num)
        self.tae = TAE(30,num_filters,num_filters)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        embe_out = self.emb(x) # (batch, sent_len, embed_dim)
        x = embe_out.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        x1 = F.relu(self.conv1(x)).squeeze(3)
        x2 = F.relu(self.conv2(x)).squeeze(3)
        x3 = F.relu(self.conv3(x)).squeeze(3)

        xa = [self.tae(x1.permute([0,2,1])),self.tae(x2.permute([0,2,1])),self.tae(x3.permute([0,2,1]))]


        x = torch.cat([i for i in xa], -1)
        x = self.dropout(x)
        logit = self.fc1(x) # (batch, target_size)
        

        return logit