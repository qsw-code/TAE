import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class topic_fusion_att_Layer(nn.Module):
    def __init__(self, channel, hidden):
        super(topic_fusion_att_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden, channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel, hidden, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        y = self.avg_pool(x.permute([0,2,1]))
        y = self.fc(y.squeeze())
        return x * y.unsqueeze(1)

class TopicAttention(nn.Module):
    def __init__(self, mid_hidden,topic_emb,hidden_unit=256):
        super(TopicAttention,self).__init__()

        self.attn_Ua = nn.Linear(topic_emb, hidden_unit)
        self.attn_Wa = nn.Linear(mid_hidden, hidden_unit, bias=False)
        self.v = nn.Linear(hidden_unit, 1, bias=False)

    def forward(self, encoder_hiddens, last_dec_hidden, masks):



        masks = torch.unsqueeze(masks, 2)  # N, 1, L

        encoder_hiddens = encoder_hiddens.unsqueeze(1)

        output_enc = self.attn_Ua(encoder_hiddens)

        output_dec = self.attn_Wa(last_dec_hidden).masked_fill(~masks, -np.inf)



        output = output_enc + output_dec
        energy = torch.tanh(output)

        attention = self.v(energy)

        r = F.softmax(attention, dim=1)

        out = r*last_dec_hidden
        out = torch.sum(out,1)

        return out

class TAE(nn.Module):
    """docstring for TAE"""
    def __init__(self, topic_num,topic_emb_len,hidden_size):
        super(TAE, self).__init__()
        self.topic_num = topic_num
        self.topic_emb = nn.Embedding(topic_num, topic_emb_len).cuda()
        nn.init.xavier_uniform_(self.topic_emb.weight)
        self.topc_att = topic_fusion_att_Layer(topic_num,hidden_size).cuda()
        self.clusterAttList = nn.ModuleList([TopicAttention(hidden_size,topic_emb_len).cuda() for _ in range(topic_num)])

    def forward(self,inputs,masks):

        docAttentionList = []
        for ind in range(self.topic_num):
            tff = self.topic_emb(torch.LongTensor([ind]).cuda())
            clusterQuery = tff.repeat(inputs.size()[0],1)
            docAttItem = self.clusterAttList[ind](clusterQuery, inputs,masks)
            docAttentionList.append(docAttItem)

        docAttentionTensor = torch.stack(docAttentionList)
        docAttentionCluster = docAttentionTensor.permute(1, 0, 2)
        
        docAttentionCluster = self.topc_att(docAttentionCluster)
        cluster_out = torch.sum(docAttentionCluster,1)

        return cluster_out
