import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import argparse
import json
import re
from collections import Counter, defaultdict

pretrain_model_path = 'bert-base-cased'
data_rate = 0.02

with open('RAMS_1.0/scorer/event_role_multiplicities.txt', encoding='utf-8') as f:
    event_in = [line.strip().split() for line in f.readlines()]

m_event_role = {}
for event_ in event_in:
    m_event_role[event_[0]] = event_[1:]

# 抽样。
with open('RAMS_1.0/scorer/event_role_multiplicities.txt', encoding='utf-8') as f:
    label = [line.strip().split()[0].split('.') for line in f.readlines()]
label_2 = []
for tem in label:
    label_2.append(tem[0] + '.' + tem[1])
set_label = set(label_2)
# 字典化
type_to_ix = {word: i for i, word in enumerate(set_label)}
data_list = []
for _ in range(0, 38):
    data_list.append([])
with open('RAMS_1.0/data/train.jsonlines', encoding='utf-8') as f:
    train_seq_in_tem = [line for line in f.readlines()]

for seq_in_ in train_seq_in_tem:
    m = eval(seq_in_)
    # [[69, 69, [["life.die.deathcausedbyviolentevents", 1.0]]]]
    event_type = m["evt_triggers"]
    tem_split = event_type[0][2][0][0].split('.')
    tem_event_type = tem_split[0] + '.' + tem_split[1]
    data_list[type_to_ix[tem_event_type]].append(seq_in_)
    assert type_to_ix[tem_event_type] < 38

len_ = 0
for tem in data_list:
    len_ += len(tem)
print('总数据量：')
print(len_)

for i in range(0, 38):
    data_list[i] = data_list[i][:int(len(data_list[i]) * data_rate) + 1]

len_ = 0
for tem in data_list:
    len_ += len(tem)
print('按比例采样数据量:')
print(len_)
train_seq_in = []
for i in range(0, 38):
    train_seq_in += data_list[i]

# train_seq_in_1  表示的是，按照采样比例采样后的集合。

f_train_da = open('train_da.json', mode='w', encoding='UTF-8')
for tem in train_seq_in:
    f_train_da.write(tem)
f_train_da.close()


# ================================================ 数据读取 ============================================
def generate_input(tem_seq_in, tem_all_sentence_1_role, tem_segment_embedding_1_role, tem_label_1_role):
    for seq_in_ in tem_seq_in:
        m = eval(seq_in_)
        seq_in_list = []  # 列表结构的句子表示
        for list_ in m['sentences']:
            seq_in_list += list_
        # [trigger_begin, trigger_end] 包含在内
        trigger_begin = m['evt_triggers'][0][0]
        trigger_end = m['evt_triggers'][0][1]
        for event in m['gold_evt_links']:
            role_begin = event[1][0]
            role_end = event[1][1]
            role = event[2][11:]
            # ========================== role的准则：==============================
            # mask
            seq_in_list_tem = []
            for list_ in m['sentences']:
                seq_in_list_tem += list_
            for i in range(0, len(seq_in_list_tem)):
                if i >= trigger_begin and i <= trigger_end:
                    seq_in_list_tem[i] = '[MASK]'
            # seg
            seq_seg = []
            for i in range(0, len(seq_in_list)):
                seq_seg.append(0)
            for i in range(m['evt_triggers'][0][0], m['evt_triggers'][0][1] + 1):
                seq_seg[i] = 1
            seq_seg = [0] + [1] + [0] + seq_seg + [0]
            # label
            seq_label = []
            for i in range(0, len(seq_in_list)):
                seq_label.append(0)
            for i in range(role_begin, role_end + 1):
                seq_label[i] = 1
            seq_label = [0] + [0] + [0] + seq_label + [0]
            # role-prompt
            inputs = '[CLS] ' + role + ' [SEP] ' + (' '.join(seq_in_list_tem)) + ' [SEP]'
            assert len(inputs.split()) == len(seq_seg)
            tem_all_sentence_1_role.append(inputs)
            tem_segment_embedding_1_role.append(seq_seg)
            tem_label_1_role.append(seq_label)


train_all_sentence_1_role = []
train_segment_embedding_1_role = []
train_label_1_role = []
generate_input(train_seq_in, train_all_sentence_1_role, train_segment_embedding_1_role, train_label_1_role)

from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForMaskedLM.from_pretrained("bert-base-cased")
model = torch.nn.DataParallel(model, device_ids=[0])
model = model.cuda()

train_feature = [tokenizer.tokenize(line) for line in train_all_sentence_1_role]
train_feature_id = [tokenizer.convert_tokens_to_ids(line) for line in train_feature]


def generate_tokenizer_num(tem_all_sentence, tem_tokenizer_num):
    for line in tem_all_sentence:
        s1 = line.split()
        s3 = tokenizer.tokenize(line)
        seq_token_len = []
        for word in s1:
            ss = tokenizer.tokenize(word)
            seq_token_len.append(len(ss))
        assert sum(seq_token_len) == len(s3)
        assert len(seq_token_len) == len(s1)
        tem_tokenizer_num.append(seq_token_len)


train_tokenizer_num = []
generate_tokenizer_num(train_all_sentence_1_role, train_tokenizer_num)


def change_seg(tem_segment_embedding, tem_segment_embedding_1, tem_tokenizer_num):
    for i in range(0, len(tem_segment_embedding)):
        tem_list = []
        for j in range(0, len(tem_segment_embedding[i])):
            for k in range(0, tem_tokenizer_num[i][j]):
                tem_list.append(tem_segment_embedding[i][j])
        tem_segment_embedding_1.append(tem_list)


train_segment_embedding_1 = []
change_seg(train_segment_embedding_1_role, train_segment_embedding_1, train_tokenizer_num)

# train_all_sentence_1_role  原始输入
# train_feature_id  id
# train_segment_embedding_1_role  修改前的seg
# train_segment_embedding_1  修改后的seg

f_da = open('train_da_mlm.json', mode='w', encoding='UTF-8')

for i in range(0, len(train_all_sentence_1_role)):
    assert len(train_feature_id[i]) == len(train_segment_embedding_1[i])
    inputs = {'input_ids': torch.LongTensor([train_feature_id[i][:512]]),
              'token_type_ids': torch.LongTensor([train_segment_embedding_1[i][:512]]),
              'attention_mask': torch.LongTensor([[1, ] * len(train_feature_id[i][:512])])}
    with torch.no_grad():
        output = model(**inputs)
    mask_token_index = (inputs['input_ids'] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    print('-')
    print(mask_token_index.tolist())
    print('-')
    if len(mask_token_index.tolist())  == 0:
        continue
    predicted_token_id = output[0][0, mask_token_index].argmax(axis=-1)
    m = {}
    m['seq_in'] = train_all_sentence_1_role[i]
    m['segment_embed'] = train_segment_embedding_1_role[i]
    m['label'] = train_label_1_role[i]
    m['mask_tokens'] = tokenizer.decode(predicted_token_id)
    if '\"' in m['mask_tokens'] or '.' in m['mask_tokens'] or '-' in m['mask_tokens'] or '\'' in m[
        'mask_tokens'] or ',' in m['mask_tokens']:
        continue
    tem_mask_tokens = m['mask_tokens'].split()
    okk = 0
    for j in range(1, len(tem_mask_tokens)):
        if tem_mask_tokens[j] == tem_mask_tokens[j - 1]:
            okk = 1
            break
    if okk == 1:
        continue
    f_da.write(str(m) + '\n')
f_da.close()
