import numpy as np
from sklearn import preprocessing
import pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve,cross_val_score
import numpy as np
from numpy import *
from sklearn.ensemble import ExtraTreesClassifier

#load train data
train_path='traindata/'
test_path='testdata/'
train_name='land_train.txt'
test_name='land_train.txt'
train_data=np.loadtxt(train_path+train_name)
test_data=np.loadtxt(test_path+test_name)
train_x=train_data[:,5:]
train_y=train_data[:,2]

test_x=test_data[:,5:]
test_y=test_data[:,2]

test_x=preprocessing.scale(test_x)
train_x=preprocessing.scale(train_x)

#build model
model=ExtraTreesClassifier(max_depth=, max_features=, min_samples_split=, n_estimators=)
scores=cross_val_score(model,train_x,train_y,cv=5,scoring='accuracy')
print(scores.mean())
model.fit(train_x,train_y)

#-----------save model--------
save_path='model/'
filename1=save_path+'model_name.sav'
pickle.dump(model,open(filename1,'wb'))
print('end')

