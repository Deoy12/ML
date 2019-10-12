from math import sqrt
import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k = 5):
    if len(data) >=k:
        warnings.warn('K is set to  a value less than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            e_d = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([e_d, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    return vote_result, confidence

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace = True)
full_data = df.astype(float).values.tolist()

test_size = 0.2
train_set = {2: [], 4:[]}
test_set = {2: [], 4:[]}
train_data = full_data[:-int(len(full_data) * test_size)]
test_data = full_data[-int(len(full_data) * test_size)]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in train_data:
    test_set[i[-1]].append(i[:-1])
correct,total = 0,0
for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k = 4)
        if group == vote:
            correct+=1
        else:
            print(confidence)
        total +=1
print('Accuracy: ', correct/total)
