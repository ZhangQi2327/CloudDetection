
'''compute cmusion matrix

labels.txt: contain label name.

predict.txt: predict_label true_label

'''

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
data_path='D:/fydata/NMC/sea_test/data/' #修改！
data_name=['sea_train_type4.txt']
model_path='D:/fydata/NMC/edition1/'

#data_path='D:/fydata/NMC/sea_test/1/testdata/testdata2/'
#data_name=['sea_all.txt']


#model_name=['et_38_sea_type0.sav','et_38_sea_type1.sav','et_38_sea_type3.sav','et_38_sea_type4.sav','final_lr_sea_38_1025_l1.sav','final_lr_sea_689_1025_l1.sav','final_et_land_38_1025.sav','final_et_land_689_1025.sav']
model_name=['final_lr_sea_38_1025_l1.sav','final_lr_sea_689_1025_l1.sav','final_et_land_38_1025.sav','final_et_land_689_1025.sav']
#model_name=['et_38_land_type1.sav','et_38_sea_type1.sav','full_et_689_land_type1.sav','full_et_689_sea_type1.sav']
test_data=np.loadtxt(data_path+data_name[0])
#test_data=np.concatenate((data1,data2,data3))
y_data=test_data[:,2]
x_data=test_data[:,5:]
print('--------------------------',np.shape(x_data))
x_data=preprocessing.scale(x_data)

#model0=joblib.load(model_path+model_name[0])
#model3=joblib.load(model_path+model_name[1])
model4=joblib.load(model_path+model_name[0])
print(np.shape(y_data))

count=0
for i in range(len(y_data)):
    if(y_data[i]==1):
           count+=1
print(count)


y_pred=model4.predict_proba(x_data)

thresholds=[0.1,0.02,0.3,0.4,0.5,0.55,0.6,0.7,0.8,0.85,0.9]   #
#thresholds=[0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]

ETS=np.zeros(len(y_data))
accuracy=np.zeros(len(y_data))
POD=np.zeros(len(y_data))
bate=np.zeros(len(y_data))
HSS=np.zeros(len(y_data))
i=0
'''
def statistics(y_data,y_pred):
    #y_pred_new = np.zeros([len(y_data),1])
    #y_pred_new[y_pred[:,1] > threshold] = 1

    a=0
    c=0
    b=0
    d=0
    
    count_0=0
    count_1=0
    count_2=0
    pred_0=0
    pred_1=0
    pred_2=0
    y_0=0
    y_1=0
    y_2=0

    for i in range(len(y_data)):
        if(y_pred_new[i]==y_data[i]):
            if(y_data[i]==0):
              count_0+=1
            elif(y_data[i]==1):
              count_1+=1
            elif(y_data[i]==2):
              count_2+=1
        if(y_pred_new[i]==0):
           pred_0+=1
        elif(y_pred_new[i]==1):
           pred_1+=1
        elif(y_pred_new[i]==2):
           pred_2+=1
        if(y_data[i]==0):
           y_0+=1
        elif(y_data[i]==1):
           y_1+=1
        elif(y_data[i]==2):
           y_2+=1
    total=len(y_data)
    son=(count_0+count_1+count_2)/total
    mother=(pred_0*y_0+pred_1*y_1+pred_2*y_2)/(total*total)
    HSS=(son-mother)/(1-mother)
    ACC=(count_0+count_1+count_2)/total
    print('ACC:',ACC)
    print('HSS:',HSS)
  
statistics(y_data,y_pred_new)    
'''
for threshold in thresholds:
    print('threshold:',threshold)
    y_pred_new = np.zeros([len(y_data),1])
    y_pred_new[y_pred[:,1] > threshold] = 1

    a=0
    c=0
    b=0
    d=0
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

'''
#--------------PLOT ACC-----------------
font1={'family':'serif','weight':'normal','size':'30'}
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30}
#sns.set_style('white')
C=[1,2,3,4,5,6,7,8,9,10]
plt.figure(1)
plt1,=plt.plot(C,ETS,linestyle='-.',linewidth=2)
plt2,=plt.plot(C,accuracy,linestyle='-',linewidth=2)
plt3,=plt.plot(C,POD,linestyle='--',linewidth=2)
plt4,=plt.plot(C,bate,linestyle=':',linewidth=2)
plt.legend(handles=[plt1, plt2,plt3,plt4], labels=['ETS','ACC','POD','b'],loc='upper right',fontsize='large',prop=font1)
plt.xticks([1,2,3,4,5,6,7,8,9,10],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],fontsize=13)
plt.yticks(fontsize=23)
plt.show()
'''
