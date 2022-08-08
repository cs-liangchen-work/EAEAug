import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7, 6'

from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import argparse
import json
import re
from collections import Counter, defaultdict
from constraints import Constraints
import scoring_utils as util

max_padding_len = 512
pretrain_model_path = '/home/liangchen/bert-base-cased'
de_bert_dem = 236
myF1 = 0.0
temF1 = 0.0
data_rate = 0.01
file_name = 'static_dict3_22.88797915891329.pkl'


parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=40)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--canshu', type=str, default='canshu.pt')
parser.add_argument('--max_acc', type=float, default=0.3)
parser.add_argument('-g', '--gold_file', type=str,
                      help='Gold file path')
parser.add_argument('-p', '--pred_file', type=str, default=None,
                      help='Predictions file path')
parser.add_argument('--reuse_gold_format', dest='reuse_gold_format',
                      default=False, action='store_true',
                      help="Reuse gold file format for pred file.")
parser.add_argument('-t', '--ontology_file', type=str, default=None,
                      help='Path to ontology file')
parser.add_argument('-cd', '--type_constrained_decoding', dest="cd",
                      default=False, action='store_true',
                      help="Use type constrained decoding" +
                      '(only possible when ontology file is given')
parser.add_argument('--do_all', dest='do_all', default=False,
                      action='store_true', help="Do everything.")
parser.add_argument('--metrics', dest='metrics', default=False,
                      action='store_true',
                      help="Compute overall p, r, f1.")
parser.add_argument('--distance', dest='distance', default=False,
                      action='store_true',
                      help="Compute p, r, f1 by distance.")
parser.add_argument('--role_table', dest='role_table', default=False,
                      action='store_true',
                      help="Compute p, r, f1 per role.")
parser.add_argument('--confusion', dest='confusion', default=False,
                      action='store_true',
                      help="Compute an error confusion matrix.")
args = parser.parse_args()
num_epoch = args.num_epoch
lr = args.lr
canshu = args.canshu
max_acc = args.max_acc





with open('./event_role_multiplicities.txt', encoding='utf-8') as f:
    event_in = [line.strip().split() for line in f.readlines()]

m_event_role = {}
for event_ in event_in:
    m_event_role[event_[0]] = event_[1:]


# 抽样。
with open('/data/liangchen/RAMS_1.0/scorer/event_role_multiplicities.txt', encoding='utf-8') as f:
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
with open('/data/liangchen/RAMS_1.0/data/train.jsonlines', encoding='utf-8') as f:
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
print('总数据：')
print(len_)

for i in range(0, 38):
    data_list[i] = data_list[i][:int(len(data_list[i]) * data_rate)+1]

len_ = 0
for tem in data_list:
    len_ += len(tem)
print('按比例采样数据:')
print(len_)
train_seq_in_1 = []
for i in range(0, 38):
    train_seq_in_1 += data_list[i]


train_seq_in_1 = []
with open('./train_da.json', encoding='utf-8') as f:
    train_seq_in_1 = [line.strip() for line in f.readlines()]
with open('/data/liangchen/RAMS_1.0/data/dev.jsonlines', encoding='utf-8') as f:
    dev_seq_in_1 = [line.strip() for line in f.readlines()]
with open('/data/liangchen/RAMS_1.0/data/test.jsonlines', encoding='utf-8') as f:
    test_seq_in_1 = [line.strip() for line in f.readlines()]


train_seq_in = []
test_seq_in = []
dev_seq_in = []

for i in range(0, len(test_seq_in_1)):
    m = eval(test_seq_in_1[i])
    result_list = m['gold_evt_links']
    pad_ = 'evt089arg02'
    trigger_begin = m['evt_triggers'][0][0]
    trigger_end = m['evt_triggers'][0][1]
    list_event = []
    for list_ in m['gold_evt_links']:  # 这里需要修改，引入一个额外的列表即可。第i 个test_seq_in 需要预测的数目
        # [[40, 40], [28, 28], "evt089arg02place"]
        list_event.append(list_[2][11:])
    c = m["evt_triggers"]  # [[35, 35, [['artifactexistence.damagedestroy.damage', 1.0]]]]
    c_s = c[0][2][0][0]  # 'artifactexistence.damagedestroy.damage'
    c_ss = m_event_role[c_s]  # 列表
    index_c_ss = 0
    while index_c_ss < len(c_ss):
        for _ in range(list_event.count(c_ss[index_c_ss]), int(c_ss[index_c_ss + 1])):
            tem_gold = [[trigger_begin, trigger_end], [0, -1], "evt089arg02" + c_ss[index_c_ss]]
            result_list.append(tem_gold)
        index_c_ss += 2
    m['gold_evt_links'] = result_list
    test_seq_in.append(str(m))


for i in range(0, len(train_seq_in_1)):
    m = eval(train_seq_in_1[i])
    result_list = m['gold_evt_links']
    pad_ = 'evt089arg02'
    trigger_begin = m['evt_triggers'][0][0]
    trigger_end = m['evt_triggers'][0][1]
    list_event = []
    for list_ in m['gold_evt_links']:  # 这里需要修改，引入一个额外的列表即可。第i 个test_seq_in 需要预测的数目
        # [[40, 40], [28, 28], "evt089arg02place"]
        list_event.append(list_[2][11:])
    c = m["evt_triggers"]  # [[35, 35, [['artifactexistence.damagedestroy.damage', 1.0]]]]
    c_s = c[0][2][0][0]  # 'artifactexistence.damagedestroy.damage'
    c_ss = m_event_role[c_s]  # 列表
    index_c_ss = 0
    while index_c_ss < len(c_ss):
        for _ in range(list_event.count(c_ss[index_c_ss]), int(c_ss[index_c_ss + 1])):
            tem_gold = [[trigger_begin, trigger_end], [0, -1], "evt089arg02" + c_ss[index_c_ss]]
            result_list.append(tem_gold)
        index_c_ss += 2
    m['gold_evt_links'] = result_list
    train_seq_in.append(str(m))

for i in range(0, len(dev_seq_in_1)):
    m = eval(dev_seq_in_1[i])
    result_list = m['gold_evt_links']
    pad_ = 'evt089arg02'
    trigger_begin = m['evt_triggers'][0][0]
    trigger_end = m['evt_triggers'][0][1]
    list_event = []
    for list_ in m['gold_evt_links']:  # 这里需要修改，引入一个额外的列表即可。第i 个test_seq_in 需要预测的数目
        # [[40, 40], [28, 28], "evt089arg02place"]
        list_event.append(list_[2][11:])
    c = m["evt_triggers"]  # [[35, 35, [['artifactexistence.damagedestroy.damage', 1.0]]]]
    c_s = c[0][2][0][0]  # 'artifactexistence.damagedestroy.damage'
    c_ss = m_event_role[c_s]  # 列表
    index_c_ss = 0
    while index_c_ss < len(c_ss):
        for _ in range(list_event.count(c_ss[index_c_ss]), int(c_ss[index_c_ss + 1])):
            tem_gold = [[trigger_begin, trigger_end], [0, -1], "evt089arg02" + c_ss[index_c_ss]]
            result_list.append(tem_gold)
        index_c_ss += 2
    m['gold_evt_links'] = result_list
    dev_seq_in.append(str(m))


'''
train_seq_in = []
with open('./RAMS_1.0/data/train.jsonlines', encoding='utf-8') as f:
    train_seq_in = [line.strip() for line in f.readlines()]
'''

# train_seq_in = train_seq_in[:8]
# test_seq_in = test_seq_in[:2]
# dev_seq_in = dev_seq_in[:2]


# ================================================ 自定义F1 ============================================

class Scorer(object):
    def __init__(self, args):
        self.role_string_mapping = {}
        self.roles = set()
        self.gold = self.read_gold_file(args.gold_file)
        if args.reuse_gold_format:
            self.pred = self.read_gold_file(args.pred_file, confidence=False)
        else:
            self.pred = self.read_preds_file(args.pred_file)
        self.constraints = Constraints(args.ontology_file)

    def get_role_label(self, role):
        if role in self.role_string_mapping:
            return self.role_string_mapping[role]
        else:
            # Each role is of the form evt###arg##role, we only want role
            role_string = re.split(r'\d+', role)[-1]
            assert (role_string == role[11:])

            self.role_string_mapping[role] = role_string
            self.roles.add(role_string)
            return role_string

    def read_gold_file(self, file_path, confidence=False):
        """
        Returns dict mapping doc_key -> (pred, arg, role)
        """

        def process_example(json_blob):
            doc_key = json_blob["doc_key"]
            gold_evt = json_blob["gold_evt_links"]
            sents = json_blob["sentences"]
            sent_map = []
            for i, sent in enumerate(sents):
                for _ in sent:
                    sent_map.append(i)

            def span_to_sent(span):
                # assumes span does not cross boundaries
                sent_start = sent_map[span[0]]
                sent_end = sent_map[span[1]]
                if sent_start != sent_end:
                    sent_end = sent_start
                assert (sent_start == sent_end)
                return sent_start

            # There should only be one predicate
            evt_triggers = json_blob["evt_triggers"]
            assert (len(evt_triggers) == 1)

            evt_trigger = evt_triggers[0]
            evt_trigger_span = util.list_to_span(evt_trigger[:2])
            evt_trigger_types = set([evt_trigger_type[0]
                                     for evt_trigger_type in evt_trigger[2]])

            gold_evt_links = [(util.list_to_span(arg[0]),
                               util.list_to_span(arg[1]),
                               self.get_role_label(arg[2])) for arg in gold_evt]
            if confidence:
                gold_evt_links = [(a, b, c, 0) for a, b, c in gold_evt_links]
            assert (all([arg[0] == evt_trigger_span
                         for arg in gold_evt_links]))
            return (doc_key, gold_evt_links, evt_trigger_types, span_to_sent)

        jsonlines = open(file_path, 'r').readlines()
        lines = [process_example(json.loads(line)) for line in jsonlines]
        file_dict = {doc_key: (evt_links, evt_trigger_types, span_to_sent)
                     for doc_key, evt_links, evt_trigger_types, span_to_sent
                     in lines}
        return file_dict

    def read_preds_file(self, file_path):
        """
        Ideally have only a single file reader
        Returns dict mapping doc_key -> (pred, arg, role)
        """

        def process_example(json_blob):
            doc_key = json_blob["doc_key"]
            pred_evt = json_blob["predictions"]
            # There should only be one predicate
            if len(pred_evt) == 0:
                return (doc_key, [], None)
            assert (len(pred_evt) == 1)
            pred_evt = pred_evt[0]
            # convention that the 0th one is the predicate span
            evt_span = util.list_to_span(pred_evt[0])
            evt_args = pred_evt[1:]
            pred_args = [(evt_span,
                          util.list_to_span(args[:2]),
                          args[2],
                          args[3])
                         for args in evt_args]
            return doc_key, pred_args, None

        jsonlines = open(file_path, 'r').readlines()
        lines = [process_example(json.loads(line)) for line in jsonlines]
        file_dict = {doc_key: (evt_links, evt_trigger_types)
                     for doc_key, evt_links, evt_trigger_types
                     in lines}
        return file_dict

    def create_role_table(self, correct, missing, overpred):
        role_table = {}
        for role in self.roles:
            c = float(correct[role])
            m = float(missing[role])
            o = float(overpred[role])
            p, r, f1 = util.compute_metrics(c, m, o)
            role_table[role] = {'CORRECT': c,
                                'MISSING': m,
                                'OVERPRED': o,
                                'PRECISION': p,
                                'RECALL': r,
                                'F1': f1}
        total_c = sum(correct.values())
        total_m = sum(missing.values())
        total_o = sum(overpred.values())
        total_p, total_r, total_f1 = util.compute_metrics(total_c,
                                                          total_m,
                                                          total_o)
        totals = {'CORRECT': total_c,
                  'MISSING': total_m,
                  'OVERPRED': total_o,
                  'PRECISION': total_p,
                  'RECALL': total_r,
                  'F1': total_f1}
        return (role_table, totals)

    def evaluate(self, constrained_decoding=True):
        self.metrics = None
        self.distance_metrics = None
        self.role_table = None
        self.confusion = None
        # Also computes confusion counters
        global_confusion = defaultdict(Counter)
        sentence_breakdowns = [{
            "correct": Counter(),
            "missing": Counter(),
            "overpred": Counter()
        } for i in range(5)]
        total_lost = 0

        global_correct = Counter()
        global_missing = Counter()
        global_overpred = Counter()
        for doc_key, (gold_structure, evt_type, span_to_sent) in self.gold.items():
            pred_structure = self.pred.get(doc_key, ([], None))[0]
            pred_structure, lost = self.constraints.filter_preds(
                pred_structure,
                evt_type,
                constrained_decoding)

            total_lost += lost
            pred_set = Counter(pred_structure)
            gold_set = Counter(gold_structure)
            assert (sum(pred_set.values()) == len(pred_structure))
            assert (sum(gold_set.values()) == len(gold_structure))
            intersection = gold_set & pred_set
            missing = gold_set - pred_set
            overpred = pred_set - gold_set
            # Update confusion and counters
            util.compute_confusion(global_confusion, intersection,
                                   missing, overpred)
            util.update(intersection, global_correct)
            util.update(missing, global_missing)
            util.update(overpred, global_overpred)
            util.update_sentence_breakdowns(intersection, missing, overpred,
                                            sentence_breakdowns, span_to_sent)
        precision, recall, f1, _ = util.compute_from_counters(global_correct,
                                                              global_missing,
                                                              global_overpred)
        distance_metrics = []
        for i in range(5):
            i_p, i_r, i_f1, counts = util.compute_from_counters(
                sentence_breakdowns[i]["correct"],
                sentence_breakdowns[i]["missing"],
                sentence_breakdowns[i]["overpred"]
            )
            distance_metrics.append((i, (i_p, i_r, i_f1), counts))
        self.metrics = {'precision': precision,
                        'recall': recall,
                        'f1': f1}
        self.distance_metrics = distance_metrics
        self.role_table = self.create_role_table(global_correct,
                                                 global_missing,
                                                 global_overpred)
        return {"role_table": self.role_table,
                "confusion": global_confusion,
                "metrics": self.metrics,
                "distance_metrics": self.distance_metrics}


def run_evaluation(args):
    """This is a separate wrapper around args so that other programs
    can call evaluation without resorting to an os-level call
    """
    scorer = Scorer(args)
    return_dict = scorer.evaluate(constrained_decoding=args.cd)
    if args.confusion or args.do_all:
        pass
        # util.print_confusion(return_dict['confusion'])
    if args.role_table or args.do_all:
        pass
        # util.print_table(*return_dict['role_table'])
    if args.distance or args.do_all:
        for (i, (p, r, f1), (gold, pred)) in return_dict['distance_metrics']:
            print(" {} & {} & {:.1f} & {:.1f} & {:.1f} \\\\ [p r f1 {} gold/{} pred. ]".format(
                i - 2, pred, p, r, f1, gold, pred))
    if args.metrics or args.do_all:
        print("Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
            return_dict['metrics']['precision'],
            return_dict['metrics']['recall'],
            return_dict['metrics']['f1']))
        global temF1
        temF1=return_dict['metrics']['f1']

    return return_dict



# ================================================ 数据读取 ============================================
# train
train_all_sentence = []
train_label = []
train_segment_embedding = []
#  注意每一个句子产生多组 用于训练的句子。
for seq_in_ in train_seq_in:
    m = eval(seq_in_)
    seq_in_list = []  # 列表结构的句子表示
    for list_ in m['sentences']:
        seq_in_list += list_

    # 获取到trigger    在句向量中。 role 和 trigger 都应该标注为 1.
    seq_seg = []
    # 产生和原始句子一样长的，
    for i in range(0, len(seq_in_list)):
         seq_seg.append(0)
    # trigger相应位置变成1
    for i in range(m['evt_triggers'][0][0], m['evt_triggers'][0][1] + 1):
        seq_seg[i] = 1
    # cls + role + sep +句子 + sep  对seq_seg的后处理。
    seq_seg = [0] + [1] + [0] + seq_seg + [0]

    for event in m['gold_evt_links']:
        seq_label = []
        for i in range(0, len(seq_in_list)):
            seq_label.append(0)
        # [[31, 31], [27, 27], "evt043arg01communicator"]  event的结构
        begin_ = event[1][0]
        end_ = event[1][1]
        event_ = event[2][11:]
        seq_label_ = seq_label
        for i in range(begin_, end_ + 1):
            seq_label_[i] = 1
        bert_seq_in_ = '[CLS] ' + event_ + ' [SEP] ' + (' '.join(seq_in_list)) + ' [SEP]'

        if begin_ == 0 and end_ == -1:
            train_all_sentence.append(bert_seq_in_)
            train_label.append([0] + [0] + [1] + seq_label_ + [0])
            train_segment_embedding.append(seq_seg)
        else:
            train_all_sentence.append(bert_seq_in_)
            train_label.append([0] + [0] + [0] + seq_label_ + [0])
            train_segment_embedding.append(seq_seg)


# test
test_all_sentence = []
test_label = []
test_segment_embedding = []
#  注意每一个句子产生多组 用于训练的句子。
for seq_in_ in test_seq_in:
    m = eval(seq_in_)
    seq_in_list = []  # 列表结构的句子表示
    for list_ in m['sentences']:
        seq_in_list += list_

    # 获取到trigger    在句向量中。 role 和 trigger 都应该标注为 1.
    seq_seg = []
    # 产生和原始句子一样长的，
    for i in range(0, len(seq_in_list)):
         seq_seg.append(0)
    # trigger相应位置变成1
    for i in range(m['evt_triggers'][0][0], m['evt_triggers'][0][1] + 1):
        seq_seg[i] = 1
    # cls + role + sep +句子 + sep  对seq_seg的后处理。
    seq_seg = [0] + [1] + [0] + seq_seg + [0]

    for event in m['gold_evt_links']:
        seq_label = []
        for i in range(0, len(seq_in_list)):
            seq_label.append(0)
        # [[31, 31], [27, 27], "evt043arg01communicator"]  event的结构
        begin_ = event[1][0]
        end_ = event[1][1]
        event_ = event[2][11:]
        seq_label_ = seq_label
        for i in range(begin_, end_ + 1):
            seq_label_[i] = 1
        bert_seq_in_ = '[CLS] ' + event_ + ' [SEP] ' + (' '.join(seq_in_list)) + ' [SEP]'

        if begin_ == 0 and end_ == -1:
            test_all_sentence.append(bert_seq_in_)
            test_label.append([0] + [0] + [1] + seq_label_ + [0])
            test_segment_embedding.append(seq_seg)
        else:
            test_all_sentence.append(bert_seq_in_)
            test_label.append([0] + [0] + [0] + seq_label_ + [0])
            test_segment_embedding.append(seq_seg)



# dev
dev_all_sentence = []
dev_label = []
dev_segment_embedding = []
#  注意每一个句子产生多组 用于训练的句子。
for seq_in_ in dev_seq_in:
    m = eval(seq_in_)
    seq_in_list = []  # 列表结构的句子表示
    for list_ in m['sentences']:
        seq_in_list += list_

    # 获取到trigger    在句向量中。 role 和 trigger 都应该标注为 1.
    seq_seg = []
    # 产生和原始句子一样长的，
    for i in range(0, len(seq_in_list)):
         seq_seg.append(0)
    # trigger相应位置变成1
    for i in range(m['evt_triggers'][0][0], m['evt_triggers'][0][1] + 1):
        seq_seg[i] = 1
    # cls + role + sep +句子 + sep  对seq_seg的后处理。
    seq_seg = [0] + [1] + [0] + seq_seg + [0]

    for event in m['gold_evt_links']:
        seq_label = []
        for i in range(0, len(seq_in_list)):
            seq_label.append(0)
        # [[31, 31], [27, 27], "evt043arg01communicator"]  event的结构
        begin_ = event[1][0]
        end_ = event[1][1]
        event_ = event[2][11:]
        seq_label_ = seq_label
        for i in range(begin_, end_ + 1):
            seq_label_[i] = 1
        bert_seq_in_ = '[CLS] ' + event_ + ' [SEP] ' + (' '.join(seq_in_list)) + ' [SEP]'

        if begin_ == 0 and end_ == -1:
            dev_all_sentence.append(bert_seq_in_)
            dev_label.append([0] + [0] + [1] + seq_label_ + [0])
            dev_segment_embedding.append(seq_seg)
        else:
            dev_all_sentence.append(bert_seq_in_)
            dev_label.append([0] + [0] + [0] + seq_label_ + [0])
            dev_segment_embedding.append(seq_seg)


# ============================================ 截取 填充 ============================================
# [CLS] Role [SEP] Sentence [SEP]
# dev_all_sentence          字符串格式
# dev_label                 列表格式
# dev_segment_embedding     列表格式

print('loading tokenizer...')

tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
train_feature = [tokenizer.tokenize(line) for line in train_all_sentence]
test_feature = [tokenizer.tokenize(line) for line in test_all_sentence]
dev_feature = [tokenizer.tokenize(line) for line in dev_all_sentence]
train_feature_id = [tokenizer.convert_tokens_to_ids(line) for line in train_feature]
test_feature_id = [tokenizer.convert_tokens_to_ids(line) for line in test_feature]
dev_feature_id = [tokenizer.convert_tokens_to_ids(line) for line in dev_feature]


# 和句子的长度，原始标签的长度保持一致 的  列表，每一个位置表示tokenizer后对应的词的数量。
train_tokenizer_num = []
test_tokenizer_num = []
dev_tokenizer_num = []

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

for line in test_all_sentence:
    s1 = line.split()
    s3 = tokenizer.tokenize(line)
    seq_token_len = []
    for word in s1:
        ss = tokenizer.tokenize(word)
        seq_token_len.append(len(ss))
    assert sum(seq_token_len) == len(s3)
    assert len(seq_token_len) == len(s1)
    test_tokenizer_num.append(seq_token_len)

for line in dev_all_sentence:
    s1 = line.split()
    s3 = tokenizer.tokenize(line)
    seq_token_len = []
    for word in s1:
        ss = tokenizer.tokenize(word)
        seq_token_len.append(len(ss))
    assert sum(seq_token_len) == len(s3)
    assert len(seq_token_len) == len(s1)
    dev_tokenizer_num.append(seq_token_len)


# 根据tokenizer，对segment embedding 和 label 等的 数据进行修正。
# 进行相应的修改。因为此时必须包含trigger，如果不包含，则处理截取的窗口。
# 需要留存截取初始位置前面元素的个数，


train_segment_embedding_1 = []
test_segment_embedding_1 = []
dev_segment_embedding_1 = []
for i in range(0, len(train_segment_embedding)):
    # train_segment_embedding[i] 是第i个句子的列表。
    tem_list = []
    for j in range(0, len(train_segment_embedding[i])):
        # train_segment_embedding[i][j] 表示原始的位置元素。
        # train_tokenizer_num[i][j] 表示存放的数目。
        for k in range(0, train_tokenizer_num[i][j]):
            tem_list.append(train_segment_embedding[i][j])
    train_segment_embedding_1.append(tem_list)
for i in range(0, len(test_segment_embedding)):
    # train_segment_embedding[i] 是第i个句子的列表。
    tem_list = []
    for j in range(0, len(test_segment_embedding[i])):
        # train_segment_embedding[i][j] 表示原始的位置元素。
        # train_tokenizer_num[i][j] 表示存放的数目。
        for k in range(0, test_tokenizer_num[i][j]):
            tem_list.append(test_segment_embedding[i][j])
    test_segment_embedding_1.append(tem_list)
for i in range(0, len(dev_segment_embedding)):
    # train_segment_embedding[i] 是第i个句子的列表。
    tem_list = []
    for j in range(0, len(dev_segment_embedding[i])):
        # train_segment_embedding[i][j] 表示原始的位置元素。
        # train_tokenizer_num[i][j] 表示存放的数目。
        for k in range(0, dev_tokenizer_num[i][j]):
            tem_list.append(dev_segment_embedding[i][j])
    dev_segment_embedding_1.append(tem_list)


train_label_1 = []
test_label_1 = []
dev_label_1 = []
for i in range(0, len(train_label)):
    # train_segment_embedding[i] 是第i个句子的列表。
    tem_list = []
    for j in range(0, len(train_label[i])):
        # train_segment_embedding[i][j] 表示原始的位置元素。
        # train_tokenizer_num[i][j] 表示存放的数目。
        for k in range(0, train_tokenizer_num[i][j]):
            tem_list.append(train_label[i][j])
    train_label_1.append(tem_list)
for i in range(0, len(dev_label)):
    # train_segment_embedding[i] 是第i个句子的列表。
    tem_list = []
    for j in range(0, len(dev_label[i])):
        # train_segment_embedding[i][j] 表示原始的位置元素。
        # train_tokenizer_num[i][j] 表示存放的数目。
        for k in range(0, dev_tokenizer_num[i][j]):
            tem_list.append(dev_label[i][j])
    dev_label_1.append(tem_list)
for i in range(0, len(test_label)):
    # train_segment_embedding[i] 是第i个句子的列表。
    tem_list = []
    for j in range(0, len(test_label[i])):
        # train_segment_embedding[i][j] 表示原始的位置元素。
        # train_tokenizer_num[i][j] 表示存放的数目。
        for k in range(0, test_tokenizer_num[i][j]):
            tem_list.append(test_label[i][j])
    test_label_1.append(tem_list)


# feature  句子的截取 和 填充
for j in range(len(train_feature_id)):
    # 将样本数据填充至长度为 max_padding_len
    i = train_feature_id[j]
    if len(i) < max_padding_len:
        train_feature_id[j].extend([0] * (max_padding_len - len(i)))
    else:
        train_feature_id[j] = train_feature_id[j][0:max_padding_len - 1] + [train_feature_id[j][-1]]


for j in range(len(test_feature_id)):
    # 将样本数据填充至长度为 max_padding_len
    i = test_feature_id[j]
    if len(i) < max_padding_len:
        test_feature_id[j].extend([0] * (max_padding_len - len(i)))
    else:
        test_feature_id[j] = test_feature_id[j][0:max_padding_len - 1] + [test_feature_id[j][-1]]

for j in range(len(dev_feature_id)):
    # 将样本数据填充至长度为 max_padding_len
    i = dev_feature_id[j]
    if len(i) < max_padding_len:
        dev_feature_id[j].extend([0] * (max_padding_len - len(i)))
    else:
        dev_feature_id[j] = dev_feature_id[j][0:max_padding_len - 1] + [dev_feature_id[j][-1]]


# seg
for j in range(len(train_segment_embedding_1)):
    # 将样本数据填充至长度为 max_padding_len
    i = train_segment_embedding_1[j]
    if len(i) < max_padding_len:
        train_segment_embedding_1[j].extend([0] * (max_padding_len - len(i)))
    else:
        train_segment_embedding_1[j] = train_segment_embedding_1[j][0:max_padding_len - 1] + [train_segment_embedding_1[j][-1]]
for j in range(len(dev_segment_embedding_1)):
    # 将样本数据填充至长度为 max_padding_len
    i = dev_segment_embedding_1[j]
    if len(i) < max_padding_len:
        dev_segment_embedding_1[j].extend([0] * (max_padding_len - len(i)))
    else:
        dev_segment_embedding_1[j] = dev_segment_embedding_1[j][0:max_padding_len - 1] + [dev_segment_embedding_1[j][-1]]
for j in range(len(test_segment_embedding_1)):
    # 将样本数据填充至长度为 max_padding_len
    i = test_segment_embedding_1[j]
    if len(i) < max_padding_len:
        test_segment_embedding_1[j].extend([0] * (max_padding_len - len(i)))
    else:
        test_segment_embedding_1[j] = test_segment_embedding_1[j][0:max_padding_len - 1] + [test_segment_embedding_1[j][-1]]


# tokenizer_num
for j in range(len(train_tokenizer_num)):
    # 将样本数据填充至长度为 512
    i = train_tokenizer_num[j]
    if len(i) < 512:
        train_tokenizer_num[j].extend([0] * (512 - len(i)))
    else:
        train_tokenizer_num[j] = train_tokenizer_num[j][0:512 - 1] + [train_tokenizer_num[j][-1]]
for j in range(len(dev_tokenizer_num)):
    # 将样本数据填充至长度为 512
    i = dev_tokenizer_num[j]
    if len(i) < 512:
        dev_tokenizer_num[j].extend([0] * (512 - len(i)))
    else:
        dev_tokenizer_num[j] = dev_tokenizer_num[j][0:512 - 1] + [dev_tokenizer_num[j][-1]]
for j in range(len(test_tokenizer_num)):
    # 将样本数据填充至长度为 512
    i = test_tokenizer_num[j]
    if len(i) < 512:
        test_tokenizer_num[j].extend([0] * (512 - len(i)))
    else:
        test_tokenizer_num[j] = test_tokenizer_num[j][0:512 - 1] + [test_tokenizer_num[j][-1]]



train_label = train_label_1
test_label = test_label_1
dev_label = dev_label_1

for j in range(len(train_label)):
    # 将样本数据填充至长度为 max_padding_len
    i = train_label[j]
    if len(i) < max_padding_len:
        train_label[j].extend([0] * (max_padding_len - len(i)))
    else:
        train_label[j] = train_label[j][0:max_padding_len - 1] + [train_label[j][-1]]

for j in range(len(test_label)):
    # 将样本数据填充至长度为 max_padding_len
    i = test_label[j]
    if len(i) < max_padding_len:
        test_label[j].extend([0] * (max_padding_len - len(i)))
    else:
        test_label[j] = test_label[j][0:max_padding_len - 1] + [test_label[j][-1]]

for j in range(len(dev_label)):
    # 将样本数据填充至长度为 max_padding_len
    i = dev_label[j]
    if len(i) < max_padding_len:
        dev_label[j].extend([0] * (max_padding_len - len(i)))
    else:
        dev_label[j] = dev_label[j][0:max_padding_len - 1] + [dev_label[j][-1]]


train_set = TensorDataset(torch.LongTensor(train_feature_id), torch.LongTensor(train_label), torch.LongTensor(train_segment_embedding_1), torch.LongTensor(train_tokenizer_num))
train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)

test_set = TensorDataset(torch.LongTensor(test_feature_id), torch.LongTensor(test_label), torch.LongTensor(test_segment_embedding_1), torch.LongTensor(test_tokenizer_num))
test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=False)

dev_set = TensorDataset(torch.LongTensor(dev_feature_id), torch.LongTensor(dev_label), torch.LongTensor(dev_segment_embedding_1), torch.LongTensor(dev_tokenizer_num))
dev_loader = DataLoader(dataset=dev_set, batch_size=16, shuffle=False)



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
        self.linear_0 = torch.nn.Linear(embedding_dim, de_bert_dem)
        self.linear_1 = torch.nn.Linear(embedding_dim, 1)
        self.linear_2 = torch.nn.Linear(embedding_dim, 1)
        self.reasoner = nn.ModuleList()
        self.reasoner.append(DynamicReasoner(de_bert_dem, 2, 0.3))
        self.reasoner.append(DynamicReasoner(de_bert_dem, 2, 0.3))

    def forward(self, tokens, seg_embedding, attention_mask):
        output = self.model(tokens, token_type_ids=seg_embedding, attention_mask=attention_mask)
        output = output[0]
        output = self.dropout(output)
        # output = self.linear_0(output)
        # for i in range(len(self.reasoner)):
        #     output = self.reasoner[i](output)
        output_1 = self.linear_1(output)
        output_2 = self.linear_2(output)
        return output_1.squeeze(-1), output_2.squeeze(-1)


loss_func = torch.nn.CrossEntropyLoss()
model = Bert()
model = torch.nn.DataParallel(model, device_ids=[0,1,])
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
max_acc = args.max_acc
model.load_state_dict(torch.load('./model_2/'+str(file_name)))
print('start trainning....')


def dev(model, dev_dataloader):
    test_loss, test_f1, n = 0.0, 0.0, 0
    model.eval()
    all_label = []
    all_prediction = []
    with torch.no_grad():
        for data, label, seg_embedding in dev_dataloader:
            out_1, out_2 = model(data.cuda(), seg_embedding.cuda(), attention_mask=(data > 0).cuda())
            # [batch_size, len]
            n += 1
            label_begin = []
            label_end = []
            for label_ in label:
                label_begin_ = max_padding_len - 1
                label_end_ = max_padding_len - 1
                for i in range(0, max_padding_len):
                    if label_[i] == 1:
                        label_begin_ = i
                        break
                for i in range(0, max_padding_len):
                    if label_[max_padding_len - 1 - i] == 1:
                        label_end_ = max_padding_len - 1 - i
                        break
                label_begin.append(label_begin_)
                label_end.append(label_end_)
            label_begin = torch.LongTensor(label_begin)
            label_end = torch.LongTensor(label_end)
            loss = loss_func(out_1.cuda(), label_begin.cuda()) + loss_func(out_2.cuda(), label_end.cuda())
            prediction_begin = out_1.argmax(dim=1).view(-1).data.cpu().numpy().tolist()
            prediction_end = out_2.argmax(dim=1).view(-1).data.cpu().numpy().tolist()
            prediction = []
            for i in range(0, len(prediction_begin)):
                prediction_tem = []
                for _ in range(0, max_padding_len):
                    prediction_tem.append(0)
                for j in range(prediction_begin[i], prediction_end[i] + 1):
                    prediction_tem[j] = 1
                prediction.extend(prediction_tem)
            label = label.view(-1).squeeze().data.cpu().numpy().tolist()
            test_loss += loss.item()
            all_label.extend(label)
            all_prediction.extend(prediction)
    test_f1 = f1_score(all_label, all_prediction, average='macro')
    return test_loss / n, test_f1


def test(model, dev_dataloader):
    test_loss, test_f1, n = 0.0, 0.0, 0
    all_label = []
    all_prediction = []
    all_prediction_list_begin = []
    all_prediction_list_end = []
    model.eval()
    with torch.no_grad():
        for data, label, seg_embedding, tokenizer_num in dev_dataloader:
            out_1, out_2 = model(data.cuda(), seg_embedding.cuda(), attention_mask=(data > 0).cuda())
            # [batch_size, len]
            n += 1
            label_begin = []
            label_end = []
            tokenizer_id = 0
            for label_ in label:
                label_begin_ = tokenizer_num[tokenizer_id][0] + tokenizer_num[tokenizer_id][1]
                label_end_ = tokenizer_num[tokenizer_id][0] + tokenizer_num[tokenizer_id][1]
                tokenizer_id += 1
                for i in range(0, max_padding_len):
                    if label_[i] == 1:
                        label_begin_ = i
                        break
                for i in range(0, max_padding_len):
                    if label_[max_padding_len - 1 - i] == 1:
                        label_end_ = max_padding_len - 1 - i
                        break
                label_begin.append(label_begin_)
                label_end.append(label_end_)
            label_begin = torch.LongTensor(label_begin)
            label_end = torch.LongTensor(label_end)
            loss = loss_func(out_1.cuda(), label_begin.cuda()) + loss_func(out_2.cuda(), label_end.cuda())
            prediction_begin = out_1.argmax(dim=1).view(-1).data.cpu().numpy().tolist()
            prediction_end = out_2.argmax(dim=1).view(-1).data.cpu().numpy().tolist()
            all_prediction_list_begin.extend(prediction_begin)
            all_prediction_list_end.extend(prediction_end)
            prediction = []
            for i in range(0, len(prediction_begin)):
                prediction_tem = []
                for _ in range(0, max_padding_len):
                    prediction_tem.append(0)
                for j in range(prediction_begin[i], prediction_end[i] + 1):
                    prediction_tem[j] = 1
                prediction.extend(prediction_tem)
            label = label.view(-1).squeeze().data.cpu().numpy().tolist()
            test_loss += loss.item()
            all_label.extend(label)
            all_prediction.extend(prediction)
    test_f1 = f1_score(all_label, all_prediction, average='macro')

    # all_prediction_list_begin 和 all_prediction_list_end 中存放的  分别是预测的开头的结尾。
    # 而且是下标，而不是第几个位置处。
    # 对 test_segment_embedding 进行修正。一方面是前面的 cls 事件
    # 另一方面是 子词的切分。
    for i in range(0, len(test_tokenizer_num)):
        # begin 进行修正。
        all = 0
        c = 0
        for n_ in test_tokenizer_num[i]:
            all += n_
            c = c + n_ - 1
            if all >= all_prediction_list_begin[i] + 1:
                break
        all_prediction_list_begin[i] = all - 1 - c - 3
        # end 进行修正
        all = 0
        c = 0
        for n_ in test_tokenizer_num[i]:
            all += n_
            c = c + n_ - 1
            if all >= all_prediction_list_end[i] + 1:
                break
        all_prediction_list_end[i] = all - 1 - c - 3
    # 但是前面 两个额外的，所以-2
    test_index = 0
    f_1 = open('3.json', mode='w', encoding='UTF-8')
    for i in range(0, len(test_seq_in)):
        m = eval(test_seq_in[i])
        result_list = []
        pad_ = 'evt089arg02'
        for _ in range(0, len(m['gold_evt_links'])):  # 这里需要修改，引入一个额外的列表即可。第i 个test_seq_in 需要预测的数目
            # [[40, 40], [28, 28], "evt089arg02place"]
            if len(m['gold_evt_links']) <= 0:
                break
            ss = [[m['gold_evt_links'][0][0][0], m['gold_evt_links'][0][0][1]], [28, 28], ""]
            ss[1][0] = all_prediction_list_begin[test_index]
            ss[1][1] = all_prediction_list_end[test_index]
            ss[2] = pad_ + (test_all_sentence[test_index].split())[1]
            if ss[1][0] <= ss[1][1] and ss[1][0] > 0 and ss[1][1] < len(test_all_sentence[test_index].split()) - 4:
                result_list.append(ss)
            test_index += 1
        m['gold_evt_links'] = result_list
        json.dump(m, f_1, ensure_ascii=False)
        f_1.write('\n')
    f_1.close()

    return_dict = run_evaluation(args)

    return test_loss / n, test_f1


for epoch in range(args.num_epoch):
    train_loss, train_f1, n = 0.0, 0.0, 0
    all_label = []
    all_prediction = []
    for data, label, seg_embedding, tokenizer_num in train_loader:
        model.train()
        out_1, out_2 = model(data.cuda(), seg_embedding.cuda(), attention_mask=(data > 0).cuda())
        # [batch_size, len]
        n += 1
        label_begin = []
        label_end = []
        tokenizer_id = 0
        for label_ in label:
            label_begin_ = tokenizer_num[tokenizer_id][0] + tokenizer_num[tokenizer_id][1]
            label_end_ = tokenizer_num[tokenizer_id][0] + tokenizer_num[tokenizer_id][1]
            tokenizer_id += 1
            for i in range(0, max_padding_len):
                if label_[i] == 1:
                    label_begin_ = i
                    break
            for i in range(0, max_padding_len):
                if label_[max_padding_len - 1 - i] == 1:
                    label_end_ = max_padding_len - 1 - i
                    break
            label_begin.append(label_begin_)
            label_end.append(label_end_)
        label_begin = torch.LongTensor(label_begin)
        label_end = torch.LongTensor(label_end)
        loss = loss_func(out_1.cuda(), label_begin.cuda()) + loss_func(out_2.cuda(), label_end.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prediction_begin = out_1.argmax(dim=1).view(-1).data.cpu().numpy().tolist()
        prediction_end = out_2.argmax(dim=1).view(-1).data.cpu().numpy().tolist()
        prediction = []
        for i in range(0, len(prediction_begin)):
            prediction_tem = []
            for _ in range(0, max_padding_len):
                prediction_tem.append(0)
            for j in range(prediction_begin[i], prediction_end[i] + 1):
                prediction_tem[j] = 1
            prediction.extend(prediction_tem)
        label = label.view(-1).squeeze().data.cpu().numpy().tolist()
        train_loss += loss.item()
        all_label.extend(label)
        all_prediction.extend(prediction)

    train_f1 = f1_score(all_label, all_prediction, average='macro')
    train_loss = train_loss / n
    test_loss, test_f1 = test(model, test_loader)

    if temF1 > myF1:
        myF1 = temF1
        # torch.save(model.module.state_dict(), './model/static_dict.pkl')
        torch.save(model.module.state_dict(), './model/static_dict' + str(epoch) + '_' + str(myF1) + '.pkl')

    print('epoch %d, train_loss %f, train_f1 %f, dev_loss %f, dev_f1 %f' %
          (epoch + 1, train_loss, train_f1, test_loss, test_f1))
    print('----------------------------')
