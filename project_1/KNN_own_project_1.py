import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

def KNN(data,predict,k):
    if len(data) >= k:
        warnings.warn('value ok k is less then total votes')

    distance = []
    for group in data:
        for feature in data[group]:
            eq_distance = np.linalg.norm(np.array(feature)-np.array(predict))
            distance.append([eq_distance,group])

    votes = [i[1] for i in sorted(distance)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k

    return vote_result, confidence

df = pd.read_csv('data_banknote_authentication.txt')

full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
test_set = {0:[],1:[]}
train_set = {0:[],1:[]}

test_data = full_data[-int(test_size*len(full_data)):]
train_data = full_data[:-int(test_size*len(full_data))]

for i in test_data:
    # print(i[-1])
    test_set[i[-1]].append(i[:-1])
for i in train_data:
    train_set[i[-1]].append(i[:-1])
correct = 0
total = 0

for group in test_set:
    # print(group)
    for feature in test_set[group]:
        vote,confidence = KNN(train_set,feature,k=5)

        if group == vote:
            correct += 1

        total += 1

print(correct/total)
