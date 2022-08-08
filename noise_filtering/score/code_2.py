'''
base
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import argparse
import json
import re
from collections import Counter, defaultdict

max_padding_len = 512
pretrain_model_path = ''


parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--max_acc', type=float, default=0.3)
args = parser.parse_args()
num_epoch = args.num_epoch
lr = args.lr
max_acc = args.max_acc


# ================================================ role集合 ==========================
with open('./event_role_multiplicities.txt', encoding='utf-8') as f:
    event_in = [line.strip().split() for line in f.readlines()]

set_role = []
for event_ in event_in:
    for tem in event_[1:]:
        if tem != '1' and tem != '2' and tem != '3' and tem != '4':
            set_role.append(tem)

import numpy as np
lists = np.unique(set_role)
set_role = list(lists)
# print(set_role)
# print(len(set_role))
#assert 1==2
set_role = []


# ================================================ 数据读取 ============================================
# train
train_all_sentence = []
train_label = []
train_segment_embedding = []

with open('./train_da_mlm.json', encoding='utf-8') as f:
    train_mlm = [line.strip() for line in f.readlines()]

for tem in train_mlm:
    tem = eval(tem)
    # seq_in segment_embed mask_tokens
    if len(tem['mask_tokens'].split()) != tem['seq_in'].split().count('[MASK]'):
        continue
    role_begin = -1
    role_end = -1
    for i in range(0, len(tem['segment_embed'])):
        if tem['seq_in'].split()[i] == '[MASK]':
            role_begin = i
            for j in range(i, len(tem['segment_embed'])):
                if tem['seq_in'].split()[j] == '[MASK]':
                    role_end = j
                    continue
                break
            break
    assert role_end - role_begin + 1 == tem['seq_in'].split().count('[MASK]')
    seq_in_list = tem['seq_in'].split()
    label = []
    for _ in range(0, len(tem['segment_embed'])):
        label.append(0)
    for i in range(role_begin, role_end + 1):
        seq_in_list[i] = tem['mask_tokens'].split()[i-role_begin]
        label[i] = 1
    train_all_sentence.append(' '.join(seq_in_list))
    train_label.append(label)
    train_segment_embedding.append(tem['segment_embed'])
    # print(train_label[-1])
    # print(train_segment_embedding[-1])
    assert len(train_label[-1]) == len(train_segment_embedding[-1]) and len(train_segment_embedding[-1]) == len(train_all_sentence[-1].split())


with open('./ace2005.json', encoding='utf-8') as f:
    train_seq_in_ace = [line.strip() for line in f.readlines()]


from sacremoses import MosesTokenizer, MosesDetokenizer
mt = MosesTokenizer(lang='en')

train_seq_in_ace = train_seq_in_ace[:]
delete_list = ['the home of the iraqi microbiologist known as dr. germ. the woman who ran iraq\'s secret biological warfare laboratory', 'rescued p.o.w. jessica lynch']


for seq_in_ in train_seq_in_ace:
     m = eval(seq_in_)
     # print(len(m['sentences']))
     # for m2 in m['sentences']:
     # print('======================================================', len(m['sentences']))
     for m2 in m['sentences']:
         sentence = m2['text']
         for tem_1 in m2['events']:
             # print(tem_1['trigger'])
             # print('trigger        ', tem_1['trigger']['text'], '----', m2['tokens'][tem_1['trigger']['start']:tem_1['trigger']['end']])
             trigger = tem_1['trigger']['text']
             # print(tem_1['arguments'])
             for tem_2 in tem_1['arguments']:
                 # print(tem_2)
                 # print('role        ', tem_2['role'])
                 # print('text        ', tem_2['text'].replace('\n', ' '))
                 if tem_2['text'].replace('\n', ' ') in delete_list:
                     continue
                 role = tem_2['role']
                 assert len(role.split())==1
                 answer = tem_2['text'].replace('\n', ' ').replace('former Gov. Douglas Wilder', 'Douglas Wilder') \
                          .replace('Iraq\'s U.N. Ambassador Mohammed Al-Douri', 'Ambassador Mohammed Al-Douri')
                 assert trigger in sentence
                 assert answer in sentence
                 # 寻找answer在m2['tokens']中的位置。
                 begin = -1
                 end = -1
                 for i in range(0, len(m2['tokens'])):
                     if m2['tokens'][i] in answer.replace(' ', ''):
                         tem_answer = m2['tokens'][i]
                         j = i
                         while tem_answer != answer.replace(' ', '') and len(tem_answer) < len(answer.replace(' ', '')):
                             j += 1
                             tem_answer += m2['tokens'][j]
                         if tem_answer == answer.replace(' ', ''):
                             begin = i
                             end = j
                             break
                 # print(str(begin), '-----', str(end))
                 assert begin != -1 and end != -1

                 bert_seq_in_ = '[CLS] ' + role.lower() + ' [SEP] ' + (' '.join(m2['tokens'])) + ' [SEP]'
                 # print(bert_seq_in_)
                 seq_seg = []
                 train_label_ = []
                 for _ in range(0, len(m2['tokens'])):
                     seq_seg.append(0)
                     train_label_.append(0)
                 for tem_i in range(tem_1['trigger']['start'], tem_1['trigger']['end']):
                     seq_seg[tem_i] = 1
                 for tem_i in range(begin, end + 1):
                     train_label_[tem_i] = 1
                 train_all_sentence.append(bert_seq_in_)
                 train_label.append([0] + [0] + [0] + train_label_ + [0])
                 train_segment_embedding.append([0] + [1] + [0] + seq_seg + [0])



with open('./event/fn1.7.conll', encoding='utf-8') as f:
    train_seq_in_conll = [line.strip() for line in f.readlines()]


role_set = set()
max_sentence_len = 0

tem_list = []
for tem_seq in train_seq_in_conll:
    if tem_seq == '':
        sentence_token = []
        sentence_label = []
        sentence_trigger = []
        for tem_seq_2 in tem_list:
            assert len(tem_seq_2.split('	')) == 15
            sentence_token.append(tem_seq_2.split('	')[1][2:-1])
            sentence_label.append(tem_seq_2.split('	')[-1])
            sentence_trigger.append(tem_seq_2.split('	')[-3])
            tem_list = []
        # print(sentence_token)
        # print(sentence_label)
        assert len(sentence_token) == len(sentence_label)
        for i in range(0, len(sentence_token)):
            if sentence_label[i].split('-')[0] == 'S':
                j = i + 1
                seq_seg = []
                train_label_ = []
                for _ in range(0, len(sentence_label)):
                    seq_seg.append(0)
                    train_label_.append(0)
                # label
                for k in range(i, j):
                    train_label_[k] = 1
                # trigger处设置为1。
                for k in range(0, len(sentence_label)):
                    if sentence_trigger[k] != '_':
                        seq_seg[k] = 1
                bert_seq_in_ = '[CLS] ' + sentence_label[i].split('-')[1] + ' [SEP] ' + (' '.join(sentence_token)) + ' [SEP]'
                train_label.append([0] + [0] + [0] + train_label_ + [0])
                train_segment_embedding.append([0] + [1] + [0] + seq_seg + [0])
                train_all_sentence.append(bert_seq_in_)
                # 额外的其他的role的引入。
                # 额外的其他的role的引入。
                for tem_role in set_role:
                    train_label.append([0] + [0] + [0] + train_label_ + [0])
                    train_segment_embedding.append([0] + [1] + [0] + seq_seg + [0])
                    train_all_sentence.append('[CLS] ' + tem_role + ' [SEP] ' + (' '.join(sentence_token)) + ' [SEP]')
                assert len(train_label[-1]) == len(train_all_sentence[-1].split()) and len(train_segment_embedding[-1]) == len(
                    train_all_sentence[-1].split())

            elif sentence_label[i].split('-')[0] == 'B':
                j = i + 1
                while j < len(sentence_label):
                    if sentence_label[j].split('-')[0] != 'I':
                        break
                    j += 1
                seq_seg = []
                train_label_ = []
                for _ in range(0, len(sentence_label)):
                    seq_seg.append(0)
                    train_label_.append(0)
                # label
                for k in range(i, j):
                    train_label_[k] = 1
                # trigger处设置为1。
                for k in range(0, len(sentence_label)):
                    if sentence_trigger[k] != '_':
                        seq_seg[k] = 1
                bert_seq_in_ = '[CLS] ' + sentence_label[i].split('-')[1] + ' [SEP] ' + (' '.join(sentence_token)) + ' [SEP]'
                train_label.append([0] + [0] + [0] + train_label_ + [0])
                train_segment_embedding.append([0] + [1] + [0] + seq_seg + [0])
                train_all_sentence.append(bert_seq_in_)
                # 额外的其他的role的引入。
                # 额外的其他的role的引入。
                for tem_role in set_role:
                    train_label.append([0] + [0] + [0] + train_label_ + [0])
                    train_segment_embedding.append([0] + [1] + [0] + seq_seg + [0])
                    train_all_sentence.append('[CLS] ' + tem_role + ' [SEP] ' + (' '.join(sentence_token)) + ' [SEP]')
                assert len(train_label[-1]) == len(train_all_sentence[-1].split()) and len(train_segment_embedding[-1]) == len(
                    train_all_sentence[-1].split())
        continue
    else:
        tem_list.append(tem_seq)


print(len(train_all_sentence))
print(set_role)


# ============================================ 截取 填充 ============================================
# [CLS] Role [SEP] Sentence [SEP]
# dev_all_sentence          字符串格式
# dev_label                 列表格式
# dev_segment_embedding     列表格式


# train_all_sentence = train_all_sentence[:32]
# train_label = train_label[:32]
# train_segment_embedding = train_segment_embedding[:32]

print('loading tokenizer...')

tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
train_feature = [tokenizer.tokenize(line) for line in train_all_sentence]
train_feature_id = [tokenizer.convert_tokens_to_ids(line) for line in train_feature]

# 和句子的长度，原始标签的长度保持一致 的  列表，每一个位置表示tokenizer后对应的词的数量。
train_tokenizer_num = []

for line in train_all_sentence:
    s1 = line.split()
    s3 = tokenizer.tokenize(line)
    seq_token_len = []
    for word in s1:
        ss = tokenizer.tokenize(word)
        seq_token_len.append(len(ss))
    assert sum(seq_token_len) == len(s3)
    assert len(seq_token_len) == len(s1)
    train_tokenizer_num.append(seq_token_len)


# 根据tokenizer，对segment embedding 和 label 等的 数据进行修正。
# 进行相应的修改。因为此时必须包含trigger，如果不包含，则处理截取的窗口。
# 需要留存截取初始位置前面元素的个数，


train_segment_embedding_1 = []
for i in range(0, len(train_segment_embedding)):
    # train_segment_embedding[i] 是第i个句子的列表。
    tem_list = []
    for j in range(0, len(train_segment_embedding[i])):
        # train_segment_embedding[i][j] 表示原始的位置元素。
        # train_tokenizer_num[i][j] 表示存放的数目。
        for k in range(0, train_tokenizer_num[i][j]):
            tem_list.append(train_segment_embedding[i][j])
    train_segment_embedding_1.append(tem_list)


train_label_1 = []
for i in range(0, len(train_label)):
    # train_segment_embedding[i] 是第i个句子的列表。
    tem_list = []
    for j in range(0, len(train_label[i])):
        # train_segment_embedding[i][j] 表示原始的位置元素。
        # train_tokenizer_num[i][j] 表示存放的数目。
        for k in range(0, train_tokenizer_num[i][j]):
            tem_list.append(train_label[i][j])
    train_label_1.append(tem_list)


# feature  句子的截取 和 填充
for j in range(len(train_feature_id)):
    # 将样本数据填充至长度为 max_padding_len
    i = train_feature_id[j]
    if len(i) < max_padding_len:
        train_feature_id[j].extend([0] * (max_padding_len - len(i)))
    else:
        train_feature_id[j] = train_feature_id[j][0:max_padding_len - 1] + [train_feature_id[j][-1]]

# seg
for j in range(len(train_segment_embedding_1)):
    # 将样本数据填充至长度为 max_padding_len
    i = train_segment_embedding_1[j]
    if len(i) < max_padding_len:
        train_segment_embedding_1[j].extend([0] * (max_padding_len - len(i)))
    else:
        train_segment_embedding_1[j] = train_segment_embedding_1[j][0:max_padding_len - 1] + [train_segment_embedding_1[j][-1]]

# tokenizer_num
for j in range(len(train_tokenizer_num)):
    # 将样本数据填充至长度为 max_padding_len
    i = train_tokenizer_num[j]
    if len(i) < max_padding_len:
        train_tokenizer_num[j].extend([0] * (max_padding_len - len(i)))
    else:
        train_tokenizer_num[j] = train_tokenizer_num[j][0:max_padding_len - 1] + [train_tokenizer_num[j][-1]]


train_label = train_label_1

for j in range(len(train_label)):
    # 将样本数据填充至长度为 max_padding_len
    i = train_label[j]
    if len(i) < max_padding_len:
        train_label[j].extend([0] * (max_padding_len - len(i)))
    else:
        train_label[j] = train_label[j][0:max_padding_len - 1] + [train_label[j][-1]]


train_set = TensorDataset(torch.LongTensor(train_feature_id), torch.LongTensor(train_label), torch.LongTensor(train_segment_embedding_1), torch.LongTensor(train_tokenizer_num))
train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=False)



import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, mem_dim, layers, dropout, self_loop = False):
        super(GraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = torch.nn.Dropout(dropout)

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()
        self.self_loop = self_loop


    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            if self.self_loop:
                AxW = AxW  + self.weight_list[l](outputs)  # self loop
            else:
                AxW = AxW

            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return out

class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on multihead attention """

    def __init__(self,mem_dim, layers, heads, dropout):
        super(MultiGraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = torch.nn.Dropout(dropout)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[:,i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out



class StructInduction(nn.Module):
    def __init__(self, sem_dim_size, sent_hiddent_size, bidirectional):#, py_version):
        super(StructInduction, self).__init__()
        self.bidirectional = bidirectional
        self.sem_dim_size = sem_dim_size
        self.str_dim_size = sent_hiddent_size - self.sem_dim_size

        self.tp_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.tp_linear.weight)
        nn.init.constant_(self.tp_linear.bias, 0)

        self.tc_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.tc_linear.weight)
        nn.init.constant_(self.tc_linear.bias, 0)

        self.fi_linear = nn.Linear(self.str_dim_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fi_linear.weight)

        self.bilinear = nn.Bilinear(self.str_dim_size, self.str_dim_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.bilinear.weight)

        self.exparam = nn.Parameter(torch.Tensor(1, 1, self.sem_dim_size))
        torch.nn.init.xavier_uniform_(self.exparam)

        self.fzlinear = nn.Linear(3 * self.sem_dim_size, 2*self.sem_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.fzlinear.weight)
        nn.init.constant_(self.fzlinear.bias, 0)

    def forward(self, input):  # batch*sent * token * hidden

        batch_size, token_size, dim_size = input.size()

        """STEP1: Calculating Attention Matrix"""
        if (self.bidirectional):
            input = input.view(batch_size, token_size, 2, dim_size // 2)
            sem_v = torch.cat((input[:, :, 0, :self.sem_dim_size // 2], input[:, :, 1, :self.sem_dim_size // 2]), 2)
            str_v = torch.cat((input[:, :, 0, self.sem_dim_size // 2:], input[:, :, 1, self.sem_dim_size // 2:]), 2)
        else:
            sem_v = input[:, :, :self.sem_dim_size]
            str_v = input[:, :, self.sem_dim_size:]

        tp = torch.tanh(self.tp_linear(str_v))  # b*s, token, h1
        tc = torch.tanh(self.tc_linear(str_v))  # b*s, token, h1
        tp = tp.unsqueeze(2).expand(tp.size(0), tp.size(1), tp.size(1), tp.size(2)).contiguous()
        tc = tc.unsqueeze(2).expand(tc.size(0), tc.size(1), tc.size(1), tc.size(2)).contiguous()


        f_ij = self.bilinear(tp, tc).squeeze()  # b*s, token , token
        if len(f_ij.shape) == 2:
            f_ij = torch.unsqueeze(f_ij, 0)
        f_i = torch.exp(self.fi_linear(str_v)).squeeze()  # b*s, token

        mask = torch.ones(f_ij.size(1), f_ij.size(1)) - torch.eye(f_ij.size(1), f_ij.size(1))
        mask = mask.unsqueeze(0).expand(f_ij.size(0), mask.size(0), mask.size(1)).cuda()
        A_ij = torch.exp(f_ij) * mask

        """STEP: Incude Latent Structure"""
        tmp = torch.sum(A_ij, dim=1)  # nan: dimension
        res = torch.zeros(batch_size, token_size, token_size).cuda()
        # tmp = torch.stack([torch.diag(t) for t in tmp])

        res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)
        L_ij = -A_ij + res  # A_ij has 0s as diagonals

        L_ij_bar = L_ij
        L_ij_bar[:, 0, :] = f_i

        LLinv = torch.inverse(L_ij_bar)

        d0 = f_i * LLinv[:, :, 0]

        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

        tmp1 = (A_ij.transpose(1, 2) * LLinv_diag).transpose(1, 2)
        tmp2 = A_ij * LLinv.transpose(1, 2)

        temp11 = torch.zeros(batch_size, token_size, 1)
        temp21 = torch.zeros(batch_size, 1, token_size)

        temp12 = torch.ones(batch_size, token_size, token_size - 1)
        temp22 = torch.ones(batch_size, token_size - 1, token_size)

        mask1 = torch.cat([temp11, temp12], 2).cuda()
        mask2 = torch.cat([temp21, temp22], 1).cuda()

        dx = mask1 * tmp1 - mask2 * tmp2

        d = torch.cat([d0.unsqueeze(1), dx], dim=1)
        df = d.transpose(1, 2)

        ssr = torch.cat([self.exparam.repeat(batch_size, 1, 1), sem_v], 1)
        pinp = torch.bmm(df, ssr)

        cinp = torch.bmm(dx, sem_v)

        finp = torch.cat([sem_v, pinp, cinp], dim=2)

        output = F.relu(self.fzlinear(finp))

        return output, df

class StructInductionNoSplit(nn.Module):
    def __init__(self, sent_hiddent_size, bidirectional):#, py_version):
        super(StructInductionNoSplit, self).__init__()
        self.bidirectional = bidirectional
        self.str_dim_size = sent_hiddent_size #- self.sem_dim_size

        self.model_dim = sent_hiddent_size

        self.linear_keys = nn.Linear(self.model_dim, self.model_dim)
        self.linear_query = nn.Linear(self.model_dim, self.model_dim)
        self.linear_root = nn.Linear(self.model_dim, 1)

    def forward(self, input):  # batch*sent * token * hidden
        batch_size, token_size, dim_size = input.size()

        key = self.linear_keys(input)
        query = self.linear_query(input)
        f_i = self.linear_root(input).squeeze(-1)
        query = query / math.sqrt(self.model_dim)
        f_ij = torch.matmul(query, key.transpose(1, 2))

        mask = torch.ones(f_ij.size(1), f_ij.size(1)) - torch.eye(f_ij.size(1), f_ij.size(1))
        mask = mask.unsqueeze(0).expand(f_ij.size(0), mask.size(0), mask.size(1)).cuda()
        A_ij = torch.exp(f_ij) * mask

        tmp = torch.sum(A_ij, dim=1)  # nan: dimension
        res = torch.zeros(batch_size, token_size, token_size).cuda()
        res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)
        L_ij = -A_ij + res  # A_ij has 0s as diagonals

        L_ij_bar = L_ij
        L_ij_bar[:, 0, :] = f_i

        LLinv = torch.inverse(L_ij_bar)

        d0 = f_i * LLinv[:, :, 0]

        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

        tmp1 = (A_ij.transpose(1, 2) * LLinv_diag).transpose(1, 2)
        tmp2 = A_ij * LLinv.transpose(1, 2)

        temp11 = torch.zeros(batch_size, token_size, 1)
        temp21 = torch.zeros(batch_size, 1, token_size)

        temp12 = torch.ones(batch_size, token_size, token_size - 1)
        temp22 = torch.ones(batch_size, token_size - 1, token_size)

        mask1 = torch.cat([temp11, temp12], 2).cuda()
        mask2 = torch.cat([temp21, temp22], 1).cuda()

        dx = mask1 * tmp1 - mask2 * tmp2

        d = torch.cat([d0.unsqueeze(1), dx], dim=1)
        df = d.transpose(1, 2)

        output = None

        return output, df

def b_inv(b_mat):
    eye = torch.rand(b_mat.size(0), b_mat.size(1), b_mat.size(2)).cuda()
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

class DynamicReasoner(nn.Module):
    def __init__(self, hidden_size, gcn_layer, dropout_gcn):
        super(DynamicReasoner, self).__init__()
        self.hidden_size = hidden_size
        self.gcn_layer = gcn_layer
        self.dropout_gcn = dropout_gcn
        self.struc_att = StructInduction(hidden_size // 2, hidden_size, True)
        self.gcn = GraphConvLayer(hidden_size, self.gcn_layer, self.dropout_gcn, self_loop=True)

    def forward(self, input):
        '''
        :param input:
        :return:
        '''
        '''Structure Induction'''
        _, att = self.struc_att(input)
        '''Perform reasoning'''
        output = self.gcn(att[:, :, 1:], input)
        return output




class Bert(torch.nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained(pretrain_model_path).cuda()  # , config=modelConfig
        embedding_dim = self.model.config.hidden_size
        self.dropout = torch.nn.Dropout(0.5)
        self.linear_0 = torch.nn.Linear(embedding_dim, 236)
        self.linear_1 = torch.nn.Linear(embedding_dim, 1)
        self.linear_2 = torch.nn.Linear(embedding_dim, 1)
        self.reasoner = nn.ModuleList()
        self.reasoner.append(DynamicReasoner(236, 2, 0.3))
        self.reasoner.append(DynamicReasoner(236, 2, 0.3))


    def forward(self, tokens, seg_embedding, attention_mask):
        output = self.model(tokens, token_type_ids=seg_embedding, attention_mask=attention_mask)
        output = output[0]
        output = self.dropout(output)
        output_1 = self.linear_1(output)
        output_2 = self.linear_2(output)
        return output_1.squeeze(-1), output_2.squeeze(-1)


loss_func = torch.nn.CrossEntropyLoss()
model = Bert()
model = torch.nn.DataParallel(model, device_ids=[0,1])
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
max_acc = args.max_acc
print('start trainning....')

def test(model, dev_dataloader):
    p_add_begin_end = []
    model.eval()
    with torch.no_grad():
        for data, label, seg_embedding, tokenizer_num in dev_dataloader:
            out_1, out_2 = model(data.cuda(), seg_embedding.cuda(), attention_mask=(data > 0).cuda())
            softmax = torch.nn.Softmax(dim=1)
            out_1 = softmax(out_1)
            out_2 = softmax(out_2)
            for i in range(0, len(out_1)):
                tem_out_1 = out_1[i]
                tem_out_2 = out_2[i]
                label_ = label[i]
                label_begin_ = tokenizer_num[i][0] + tokenizer_num[i][1]
                label_end_ = tokenizer_num[i][0] + tokenizer_num[i][1]
                for i in range(0, max_padding_len):
                    if label_[i] == 1:
                        label_begin_ = i
                        break
                for i in range(0, max_padding_len):
                    if label_[max_padding_len - 1 - i] == 1:
                        label_end_ = max_padding_len - 1 - i
                        break
                p_add_begin_end.append((tem_out_1[label_begin_] + tem_out_2[label_end_]).cpu().item())

    return p_add_begin_end



print('---------------------------------')
mm ='./static_dict2_34.980124929017606.pkl'
model.load_state_dict(torch.load(mm))
p = test(model, train_loader)
print(len(p))
print(p)
