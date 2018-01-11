import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn import preprocessing, cross_validation
import pickle

df = pd.read_csv('bank1.csv')
df.drop(['"month"'], 1, inplace=True)

df.convert_objects(convert_numeric=True)
df.dropna(0,inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_val = {}

        def convert_to_int(val):
            return text_digit_val[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_content = df[column].values.tolist()
            unique_elements = set(column_content)

            x = 0
            for unique in unique_elements:
                if unique not in text_digit_val:
                    text_digit_val[unique] = x
                    x += 1

            df[column]= list(map(convert_to_int, df[column]))
    return df

df = handle_non_numerical_data(df)
# print(df.head())

x = np.array(df.drop(['"y"'],1).astype(float))
y = np.array(df['"y"'])

x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.2)

########################################################################
clf = neighbors.KNeighborsClassifier(n_jobs=-1)
clf.fit(x_train,y_train)

with open('KNN_code_3_1.pickle','wb') as f:
    pickle.dump(clf,f)
########################################################################

pickle_in = open('KNN_code_3_1.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test,y_test)
print(accuracy)
