import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random
from collections import defaultdict

## link: https://link.springer.com/chapter/10.1007/978-3-319-44636-3_4

def KNN(data,predict,k):
    if len(data) < k:
        warnings.warn('value of k is less then total votes')

    distance = []
    for group in data:
        for feature in data[group]:
            eq_distance = np.linalg.norm(np.array(feature)-np.array(predict))
            distance.append([eq_distance,group])

    votes = [i[1] for i in sorted(distance)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k

    return vote_result, confidence

def handle_non_numerical_data(df):
    columns = df.columns.values
    # print(columns)

    for column in columns:
        text_digit_val = {}
        def convert_to_int(val):
            return text_digit_val[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_content = df[column].values.tolist()
            unique_element = set(column_content)

            x = 0
            for unique in unique_element:
                if unique not in text_digit_val:
                    text_digit_val[unique] = x
                    x+= 1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = pd.read_csv('data.csv')
df.drop(['RecordID'], 1, inplace= True)
# print(df.head())
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

df = handle_non_numerical_data(df)

full_data = df.astype(float).values.tolist()
random.shuffle(full_data)
# print(full_data)
test_size = 0.2
test_set = defaultdict(list)
train_set = defaultdict(list)

test_data = full_data[-int(test_size*len(full_data)):]
train_data = full_data[:-int(test_size*len(full_data))]

for i in test_data:
    # print(i[-1])
    test_set[i[-1]].append(i[:-1])
for i in train_data:
    train_set[i[-1]].append(i[:-1])
correct = 0
total = 0
# print(train_set[9])

for group in test_set:
    # print(group)
    for feature in test_set[group]:
        # len(train_set)
        vote,confidence = KNN(train_set,feature,k=5)
##        print(group)

        if group == vote:
            correct += 1

        total += 1

print(correct/total)
print(confidence)

