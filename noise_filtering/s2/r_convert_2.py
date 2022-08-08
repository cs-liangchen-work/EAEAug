'''
只要第一个，只是当做一个分数而已。
'''
import torch
with open('./rr_1.txt', mode='r', encoding='utf-8') as f2:
    seq_in_2 = [ line.strip() for line in f2.readlines()]

seq_in_2 = seq_in_2[6][1:-1].split(",")

# 0.006214780267328024   列表
# tensor(0.0062)         tensor

p_ = []
p_2 = [float(n) for n in seq_in_2]
p_2 = torch.FloatTensor(p_2)

pp_2 = []
for tem in p_2:
    pp_2.append(tem.item())

print(pp_2[0])



pp = torch.FloatTensor(pp_2)

f = open('./data_select.txt',mode='w',encoding='utf-8')

for i in range(0, len(pp_2)):
    if i in torch.topk(pp, 5000).indices and i not in torch.topk(pp, 0).indices:
        f.write('1' + '\n')
    else:
        f.write('0' + '\n')

f.close()

