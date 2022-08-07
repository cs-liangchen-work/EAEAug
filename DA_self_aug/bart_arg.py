      import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import argparse
import json
import re
from collections import Counter, defaultdict
from transformers import BartForConditionalGeneration, BartTokenizer

pretrain_model_path = ''
data_rate = 0.01

with open('./RAMS_1.0/scorer/event_role_multiplicities.txt', encoding='utf-8') as f:
    event_in = [line.strip().split() for line in f.readlines()]

m_event_role = {}
for event_ in event_in:
    m_event_role[event_[0]] = event_[1:]

# 抽样。
with open('./RAMS_1.0/scorer/event_role_multiplicities.txt', encoding='utf-8') as f:
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
with open('./RAMS_1.0/data/train.jsonlines', encoding='utf-8') as f:
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
def generate_input(tem_seq_in, tem_all_sentence_1_role, tem_segment_embedding_1_role):
    for seq_in_ in tem_seq_in:
        m = eval(seq_in_)
        seq_in_list = []  # 列表结构的句子表示
        for list_ in m['sentences']:
            seq_in_list += list_
        # [trigger_begin, trigger_end] 包含在内
        trigger_begin = m['evt_triggers'][0][0] + 3
        trigger_end = m['evt_triggers'][0][1] + 3
        for event in m['gold_evt_links']:
            role_begin = event[1][0]
            role_end = event[1][1]
            role = event[2][11:]
            # print('---------------------------')
            # print(role_begin)
            # print(role_end)
            # ========================== role的准则：==============================
            # 如果为n：n-1, n, n+1
            # ===== n-1 =====
            # mask
            seq_in_list_tem_2 = []
            for list_ in m['sentences']:
                seq_in_list_tem_2 += list_
            # print(seq_in_list_tem_2[m['evt_triggers'][0][0]:m['evt_triggers'][0][1]+1])
            # print(' '.join(seq_in_list_tem_2))
            for i in range(0, len(seq_in_list_tem_2)):
                if i >= role_begin and i <= role_end:
                    seq_in_list_tem_2[i] = '<mask>'
            # seg
            seq_seg_2 = []
            for i in range(0, len(seq_in_list_tem_2)):
                seq_seg_2.append(0)
            for i in range(m['evt_triggers'][0][0], m['evt_triggers'][0][1] + 1):
                seq_seg_2[i] = 1
            seq_seg_2 = [0] + [1] + [0] + seq_seg_2 + [0]
            # 调整。
            for i in range(0, role_end - role_begin):
                del seq_seg_2[role_begin + 4]
                del seq_in_list_tem_2[role_begin + 1]
            # role-prompt
            inputs_2 = '[CLS] ' + role + ' [SEP] ' + (' '.join(seq_in_list_tem_2)) + ' [SEP]'
            # print(inputs_2)
            # print(seq_seg_2)
            assert len(inputs_2.split()) == len(seq_seg_2)
            # 删除至一个<mask>
            tem_all_sentence_1_role.append(inputs_2)
            tem_segment_embedding_1_role.append(seq_seg_2)




train_all_sentence_1_role = []
train_segment_embedding_1_role = []
generate_input(train_seq_in, train_all_sentence_1_role, train_segment_embedding_1_role)


from transformers import BartForConditionalGeneration, BartTokenizer
import torch

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model = torch.nn.DataParallel(model, device_ids=[0])
model = model.cuda()



f_da = open('train_da_mlm.json', mode='w', encoding='UTF-8')

for i in range(0, len(train_all_sentence_1_role)):
    inputs = ' '.join(train_all_sentence_1_role[i].split()[3:-1])
    print(inputs)
    input_ids = tokenizer(inputs, return_tensors="pt")["input_ids"]
    output = model(input_ids)[0]
    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    probs = output[0, masked_index].softmax(dim=0)
    values, predictions = probs.topk(3)
    predictions_list = tokenizer.decode(predictions).split()

    for j in range(0, len(predictions_list)):
        m = {}
        m['seq_in'] = train_all_sentence_1_role[i]
        m['segment_embed'] = train_segment_embedding_1_role[i]
        m['mask_tokens'] = predictions_list[j]
        f_da.write(str(m) + '\n')


f_da.close()



