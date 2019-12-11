from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from matplotlib import colors
from numpy import *

import numpy as np
from sklearn.externals import joblib

#------------load testdata------------
data_path='testdata/data/' 
data_name=['data_name.txt']
model_path='model/'
model_name=['model_name.sav]
test_data=np.loadtxt(data_path+data_name[0])
y_data=test_data[:,2]
x_data=test_data[:,5:]
x_data=preprocessing.scale(x_data)
model=joblib.load(model_path+model_name[0])

y_pred=model.predict_proba(x_data)

thresholds=[0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.7,0.8,0.9]   

def statistics(y_data,y_pred):  
  for threshold in thresholds:
    ETS=np.zeros(len(y_data))
    accuracy=np.zeros(len(y_data))
    POD=np.zeros(len(y_data))
    FARate=np.zeros(len(y_data))
    HSS=np.zeros(len(y_data))
    a=0
    c=0
    b=0
    d=0
    j=0
    print('threshold:',threshold)
    y_pred_new = np.zeros([len(y_data),1])
    y_pred_new[y_pred[:,1] > threshold] = 1
    for i in range(len(x_data)):
     if(y_pred_new[i]==y_data[i]):
        if(y_data[i]==1):
           a+=1
        else:
           d+=1
     elif(y_pred_new[i]!=y_data[i]):
        if(y_data[i]==0):
           b+=1
        elif(y_data[i]==1):
           c+=1
    total=(a+c+b+d)
    print('a:   ',a)
    print('c:   ',c)
    print('b:    ',b)
    print('d:     ',d)
    print('total:  ',total)
    a_random=((a+c)*(a+b))/(a+c+b+d)
    ETS[j]=((a-a_random)/(a+c+b-a_random))
    HSS[j]=2*(a*d-b*c)/((a+c)*(d+c)+(a+b)*(d+b))
    print('HSS:    ',HSS[j])
    print('ETS:    ',ETS[j])
    accuracy[j]=((a+d)/(a+d+b+c))
    print('ACC:    ',accuracy[j])
    POD[i]=a/(a+c)
    bate[i]=b/(b+a)
    print('POD:    ',POD[j])
    print('bate   ',bate[j])
    j=j+1


