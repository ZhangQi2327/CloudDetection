import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.model_selection import learning_curve

from sklearn.utils import shuffle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle

#----------------------train_data---------------
#load train data
train_path='D:/fydata/NMC/sea_train/data/'
name1='sea_train_type1.txt'
name2='sea_train_type2.txt'
train_data=np.loadtxt(train_path+name2)

train_data=preprocessing.scale(train_data)

x_train=train_data[:,4:]
y_train=train_data[:,1]


le=LabelEncoder()
y_train=le.fit_transform(y_train)#类标整数化


def plot_validation_curve(x,y):
    model=LogisticRegression(random_state=1,penalty='l2',solver='lbfgs',max_iter=600)
    x_train,y_train=shuffle(x,y)
    print('shape of x_train.{}'.format(np.shape(x_train)))

    font1={'family':'serif','weight':'normal','size':'28'}

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 35,
            }
    param_range=[0.001,0.01,0.1,1,10,100,1000]
    #10折，验证正则化参数C
    C=[1,2,3,4,5,6,7]
    train_scores,test_scores =validation_curve(estimator=model,X=x_train,y=y_train,param_name='C',param_range=param_range,cv=10,scoring='accuracy')
    #统计结果
    train_mean= np.mean(train_scores,axis=1)
    train_std = np.std(train_scores,axis=1)
    test_mean =np.mean(test_scores,axis=1)
    test_std=np.std(test_scores,axis=1)
    plt.plot(C,train_mean,color='red',marker='^',markersize=5,label='Training set')
    plt.fill_between(C,train_mean+train_std,train_mean-train_std,alpha=0.2,color='red')
    plt.plot(C,test_mean,color='green',linestyle='--',marker='D',markersize=5,label='Test set')
    plt.fill_between(C,test_mean+test_std,test_mean-test_std,alpha=0.2,color='green')

    #plt.xscale('log')
    plt.xlabel('Parameter C',font2)
    plt.ylabel('Auc',font2)
    plt.legend(loc='lower right',fontsize='large',prop=font1)
    plt.xticks(C,[0.001,0.01,0.1,1,10,100,1000],fontsize=20)
    plt.yticks([0.84,0.86,0.88,0.90,0.92,0.94,0.96,0.98,1.0],[0.84,0.86,0.88,0.90,0.92,0.94,0.96,0.98,1.0],fontsize=20)
    plt.grid(color='black',linewidth=1,linestyle='-.')
    plt.title('Model:lr(38 channels)          penalty:L2',fontsize=30)  
    plt.show()
plot_validation_curve(x_train,y_train)
