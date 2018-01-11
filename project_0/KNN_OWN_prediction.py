from math import sqrt
import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

def KNN(data,predict,k):
    if len(data) >= k:
        warnings.warn('k is set to a value less then totel voting groups!')

    distance = []
    for group in data:
        for features in data[group]:
            # eq_distance = sqrt((group[0]-features[0])**2+(group[1]-features[1])**2)
            # eq_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            eq_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distance.append([eq_distance,group])
    votes = [i[1] for i in sorted(distance)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    # print(vote_result)
    return vote_result,confidence

accuary = []
for _ in range(1):
    df = pd.read_csv('breast-cancer-dataseta.txt')
    df.replace('?',-99999,inplace=True)
    df.drop(['id'],1,inplace=True)

    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)


    test_size = 0.2
    test_set = {2:[],4:[]}
    train_set = {2:[],4:[]}

    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    [test_set[i[-1]].append(i[:-1]) for i in test_data]
    [train_set[i[-1]].append(i[:-1]) for i in train_data]
    # print(test_set[4])

    correct = 0
    total = 0
    c = 0
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = KNN(train_set, data,k=5)
            if group == vote:
                correct += 1
            # else:
            #     print(confidence)
            total += 1
    # print("Accuary:", correct/total)
    accuary.append(correct/total)

print(sum(accuary)/len(accuary))