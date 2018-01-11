import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import pickle

db = pd.read_csv('data_banknote_authentication.txt')

x = np.array(db.drop(['class'],1))
y = np.array(db['class'])

x_train,x_test,y_train,y_test =cross_validation.train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier(n_jobs=-1)
clf.fit(x_train,y_train)

with open('KNN_project_1.pickle','wb') as f:
    pickle.dump(clf,f)

pickle_in = open('KNN_project_1.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test,y_test)
print(accuracy)

example = np.array([[2,4,-2,5],
                    [-8,7,1,-6],
                    [-12,5,-3,-2],])
example = example.reshape(len(example),-1)
prediction = clf.predict(example)
print(prediction)
