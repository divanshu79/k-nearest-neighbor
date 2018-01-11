import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import pickle
# accuracy = []
df = pd.read_csv('breast-cancer-dataseta.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'],1,inplace=True)

x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=.2)

clf = neighbors.KNeighborsClassifier(n_jobs=-1)
clf.fit(x_train,y_train)

with open('KNN_code1.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('KNN_code1.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test,y_test)

print(accuracy)

example_measure = np.array([[4,2,1,1,1,2,3,2,1],[8,9,10,7,1,4,3,2,2]])
example_measure = example_measure.reshape(len(example_measure),-1)
pridiction = clf.predict(example_measure)

    # accuracy.append(accuracy)
# print(sum(accuracy)/len(accuracy))
