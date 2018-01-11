from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter

style.use('fivethirtyeight')

db = {'k':[[1, 2],[2, 3],[3, 2]], 'r':[[5, 6],[6, 7],[7, 6]]}
new_pnt = [1,7]

# [[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in db[i]] for i in db]
# plt.scatter(new_pnt[0],new_pnt[1])
# plt.show()

# eq_distavce = sqrt((a[0]-p[0])**2+(a[1]-p[1])**2)
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
    # print(vote_result)
    return vote_result

k = int(input())
result = KNN(db,new_pnt,k)
[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in db[i]] for i in db]
plt.scatter(new_pnt[0],new_pnt[1], color=result)
plt.show()

print(result)
