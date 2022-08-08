'''
只要第一个，只是当做一个分数而已。
'''

with open('./r_1.txt', mode='r', encoding='utf-8') as f1:
    seq_in = [ line.strip() for line in f1.readlines()]

with open('./r_2.txt', mode='r', encoding='utf-8') as f2:
    seq_in_2 = [ line.strip() for line in f2.readlines()]

print(len(seq_in))
seq_in = seq_in[6][1:-1].split(",")
seq_in_2 = seq_in_2[6][1:-1].split(",")

p = []
import torch
p = [float(n) for n in seq_in]
p = torch.FloatTensor(p)

print(p.shape)
pp = []
for tem in p:
    pp.append(tem.item())

print(pp[0])
print(p[0])
# 0.006214780267328024   列表
# tensor(0.0062)         tensor

p_ = []
p_2 = [float(n) for n in seq_in_2]
p_2 = torch.FloatTensor(p_2)

pp_2 = []
for tem in p_2:
    pp_2.append(tem.item())
print(pp[0])
print(pp_2[0])

n=0
for i in range(0,len(pp)):
    if pp[i] < pp_2[i]:
        n+=1
print(n)





n = 0
f = open('./data_select.txt',mode='w',encoding='utf-8')

for i in range(0, len(pp_2)):
    if pp[i]<pp_2[i]:
        f.write('1' + '\n')
        n+=1
    else:
        f.write('0' + '\n')
print(n)
f.close()









'''
# 排序后的分数
result_list = sorted(pp, key=lambda x: x*-1)
print('---------------')
aa = 0
while True:
    if aa>=len(result_list):
        break
    print(aa,' : ', result_list[aa])
    aa += 5000
'''
