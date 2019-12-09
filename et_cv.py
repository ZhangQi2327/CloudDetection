import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score
#----------------------train_data---------------
#load train data

#load train data
train_path='D:/fydata/NMC/sea_train/data/'
test_path='D:/fydata/NMC/sea_test/data/'
train_name='sea_train_type0.txt'
test_name='sea_train_type0.txt'
train_data=np.loadtxt(train_path+train_name)
test_data=np.loadtxt(test_path+test_name)
train_x=train_data[:,5:]
train_y=train_data[:,2]

test_x=test_data[:,5:]
test_y=test_data[:,2]
np.random.shuffle(train_data)

test_x=preprocessing.scale(test_x)
train_x=preprocessing.scale(train_x)





params={'n_estimators':[170,180,200],'max_depth':[40,50,60],'max_features':[10,16,24]}
model=ExtraTreesClassifier(min_samples_split=2)
gridcv=GridSearchCV(model,params,scoring='accuracy',cv=5,verbose=6)
grid_result=gridcv.fit(train_x,train_y)
print("Best: %f using %s" % (grid_result.best_score_,gridcv.best_params_))

#Best: 0.845870 using {'max_depth': 45, 'max_features': 18, 'n_estimators': 160}

#------------------------type 2----38------- sea -----------------
#Best: 0.901477 using {'max_depth': 20, 'max_features': 25, 'min_samples_split': 5, 'n_estimators': 130}
#Best: 0.902027 using {'max_depth': 30, 'max_features': 20, 'min_samples_split': 2, 'n_estimators': 130}
#------------------------type 2---------------689------- sea---------
#Best: 0.901752 using {'max_depth': 30, 'max_features': 20, 'min_samples_split': 2, 'n_estimators': 130}
#Best: 0.901752 using {'max_depth': 20, 'max_features': 20, 'min_samples_split': 2}

#------------------type 1 ---------38 -----------sea-------------------
#Best: 0.929029 using {'max_depth': 30, 'max_features': 25, 'min_samples_split': 5, 'n_estimators': 120}
#----------type 1 -----------689--------------sea
#Best: 0.927992 using {'max_depth': 30, 'max_features': 30, 'min_samples_split': 2, 'n_estimators': 120}


