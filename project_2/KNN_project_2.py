import numpy as np
from sklearn import neighbors
from sklearn import preprocessing, cross_validation
import  pandas as pd
import pickle
# import csv

# spamReader = csv.reader(open('data.csv', newline=''), delimiter=' ', quotechar='|')
#
# for row in spamReader:
#     print(', '.join(row))

df = pd.read_csv('data.csv')
df.drop(['RecordID'], 1, inplace= True)
# print(df.head())
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

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

df = handle_non_numerical_data(df)

# print(df.head())

x = np.array(df.drop(['Species'],1).astype(float))
y = np.array(df['Species'])

x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)

with open('KNN_code2.pickle','wb') as f:
    pickle.dump(clf, f)

pickle_in = open('KNN_code2.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test,y_test)
print(accuracy)
