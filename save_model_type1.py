import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
from sklearn.model_selection import learning_curve

from sklearn.utils import shuffle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve,cross_val_score
import numpy as np
from numpy import *
from sklearn.model_selection import validation_curve
from sklearn.ensemble import ExtraTreesClassifier

#load train data
train_path='D:/fydata/NMC/land_train/data/'
test_path='D:/fydata/NMC/land_test/data/'
train_name='land_train_type1.txt'
test_name='land_train_type1.txt'
train_data=np.loadtxt(train_path+train_name)
test_data=np.loadtxt(test_path+test_name)
train_x=train_data[:,5:]
train_y=train_data[:,2]

test_x=test_data[:,5:]
test_y=test_data[:,2]


test_x=preprocessing.scale(test_x)
train_x=preprocessing.scale(train_x)

print(np.shape(train_y))

model=ExtraTreesClassifier(max_depth=40, max_features=16, min_samples_split=2, n_estimators=140)
scores=cross_val_score(model,train_x,train_y,cv=10,scoring='accuracy')
print(scores.mean())
model.fit(train_x,train_y)

'''
yes=0
for i in range(len(test_y)):
  if(test_y[i]==prediction[i]):
    yes+=1
print(yes)
acc=yes/len(test_y)
print(acc)
'''
'''
count=0
for i in range(len(test_y)):
    if(prediction[i]==test_y[i]):
       count+=1
acc=count/len(test_y)
print(acc)
'''


thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]   #
#thresholds=[0.15,0.16,0.17,0.18,0.19,0.20,0.22,0.23,0.24,0.25]
#thresholds=[0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.80,0.81,0.82,0.83,0.84,0.85,0.86]
pred=model.predict_proba(test_x)
print(np.shape(pred))
ETS=[]
accuracy=[]
POD=[]
FARate=[]

def statistics(y_data,y_pred):
  ETS=np.zeros(len(y_data))
  accuracy=np.zeros(len(y_data))
  POD=np.zeros(len(y_data))
  bate=np.zeros(len(y_data))
  HSS=np.zeros(len(y_data))
  i=0
  for threshold in thresholds:
    y_pred_new = np.zeros([len(y_data),1])
    y_pred_new[y_pred[:,1] > threshold] = 1
    a=0
    c=0
    b=0
    d=0
    print('threshold:  ',threshold)
    for i in range(len(y_data)):
     if(y_pred_new[i]==y_data[i]):
        if(y_data[i]==1):
           a+=1
        else:
           d+=1
     elif(y_pred_new[i]!=y_data[i]):
        if(y_data[i]==0):
           b+=1
        else:
           c+=1
    total=(a+c+b+d)
    print('a:   ',a)
    print('c:   ',c)
    print('b:    ',b)
    print('d:     ',d)
    print('total:  ',total)
    a_random=((a+c)*(a+b))/(a+c+b+d)
    ETS[i]=((a-a_random)/(a+c+b-a_random))
    
   # son=(a+d)/total
   # mother=(a+b)*(c+d)/(total*total)
   # HSS[i]=((son-mother)/(1-mother))
    HSS[i]=2*(a*d-b*c)/((a+c)*(d+c)+(a+b)*(d+b))
    
    print('HSS:    ',HSS[i])
    print('ETS:    ',ETS[i])
    accuracy[i]=((a+d)/(a+d+b+c))
    print('ACC:    ',accuracy[i])
    POD[i]=a/(a+c)
    bate[i]=b/(b+a)
    print('POD:    ',POD[i])
    print('bate   ',bate[i])
    i=i+1
    
statistics(test_y,pred)

#-----------save model--------
save_path='D:/fydata/NMC/edition_2/'
filename9=save_path+'et_38_land_type1.sav'
pickle.dump(model,open(filename9,'wb'))
print('end')

