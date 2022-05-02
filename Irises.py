# -*- coding =utf-8 -*-
# @Time : 2022-05-02 14:23
# @Author : Elon
# @File : Irises.py
# @Software : PyCharm
# -*- coding =utf-8 -*-
# @Time : 2022-05-02 11:16
# @Author : Elon
# @File : Irises.py
# @Software : PyCharm
from sklearn import datasets
import pandas as pd
dataset = datasets.load_iris()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df)
print(type(df))
df['target'] = dataset['target']
df.to_csv('./data/Irises.csv', index=None)


