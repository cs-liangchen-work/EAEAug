'''
只要第一个，只是当做一个分数而已。
'''

with open('./r_1.txt', mode='r', encoding='utf-8') as f2:
    seq_in = [ line.strip() for line in f2.readlines()]

print(len(seq_in))
seq_in = seq_in[5][1:-1].split(",")

p = []

import torch
p = [float(n) for n in seq_in]
p = torch.FloatTensor(p)


cc = 0.3
n = 0
for pp in p:
    if pp>=cc:
        n+=1
print(n)

print(p.shape)
pp = []
for tem in p:
    pp.append(tem.item())
print(len(pp))

print(pp[0])
print(p[0])

pp = torch.FloatTensor(pp)

f = open('./data_select.txt',mode='w',encoding='utf-8')

for i in range(0, 34219):
    if i in torch.topk(pp, 3000).indices:
        f.write('1' + '\n')
    else:
        f.write('0' + '\n')

f.close()   
