# -*- coding =utf-8 -*-
# @Time : 2022-04-26 20:01
# @Author : Elon
# @File : bpnet.py
# @Software : PyCharm
# -*- coding =utf-8 -*-
# @Time : 2022-04-25 8:51
# @Author : Elon
# @File : bpnet.py
# @Software : PyCharm
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(4, 20)   # 定义隐藏层网络

        self.out = torch.nn.Linear(20, 3)   # 定义输出层网络

    def forward(self, x):
        x = F.relu(self.hidden(x))      # 隐藏层的激活函数,采用relu,也可以采用sigmod,tanh
        x = self.out(x)                   # 输出层不用激活函数
        return x


