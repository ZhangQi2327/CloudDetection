import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesClassifier
#----------------------train_data---------------
train_path='train_data/'
train_name='data_name.txt'
train_data=np.loadtxt(train_path+train_name)
train_x=train_data[:,5:]
train_y=train_data[:,2]
train_x,train_y=np.random.shuffle(train_x,train_y)
train_x=preprocessing.scale(train_x)

params={'n_estimators':[100,120,130,140,150,170,180,200],'max_depth':[5,10,15,20,25,30,35,40,45,50,55,60],'max_features':[3,6,9,12,15,18,21,24],'min_samples_split':[2,3,4,5,6,7,8,9],'min_samples_leaf':[1,2,3,4,5,6,7,8,9]}
model=ExtraTreesClassifier()
gridcv=GridSearchCV(model,params,scoring='accuracy',cv=5,verbose=6)
grid_result=gridcv.fit(train_x,train_y)
print("Best: %f using %s" % (grid_result.best_score_,gridcv.best_params_))



