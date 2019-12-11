import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesClassifier
import pickle 


#----------------------train_data---------------
#load train data
train_path='D:/fydata/NMC/sea_train/data/'
test_path='D:/fydata/NMC/sea_test/data/'
train_name='full_sea_train_type0.txt'
test_name='full_sea_train_type0.txt'
train_data=np.loadtxt(train_path+train_name)
test_data=np.loadtxt(test_path+test_name)
train_x=train_data[:,5:]
train_y=train_data[:,2]

test_x=test_data[:,5:]
test_y=test_data[:,2]

test_x=preprocessing.scale(test_x)
train_x=preprocessing.scale(train_x)
'''
model=ExtraTreesClassifier(max_depth=30, max_features=30, min_samples_split=2, n_estimators=140)
model.fit(train_x,train_y)

save_path='D:/fydata/NMC/edition_2/'
filename9=save_path+'lr_train_type0.sav'
pickle.dump(model,open(filename9,'wb'))
print('end')
'''
def test_learning_curve(X,y):

    ### 加载数据
    X_shuffle, y_shuffle = shuffle(X, y)
    #### 获取学习曲线 ######
    train_sizes=[10000,12000,14000,16000,18000,20000,22000]
    model=ExtraTreesClassifier()
    #model=LogisticRegression(C=10,penalty='l2',solver='lbfgs',max_iter=700,multi_class='auto')
    abs_trains_sizes,train_scores, test_scores = learning_curve(model,X_shuffle, y_shuffle,cv=5, scoring="accuracy",train_sizes=train_sizes)
    ###### 对每个 C ，获取 10 折交叉上的预测得分上的均值和方差 #####
    
    print('train_sizes',abs_trains_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ####### 绘图 ######
    font1={'family':'serif','weight':'normal','size':'28'}
    font1={'family':'serif','weight':'normal','size':'28'}

    font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 40,
            }
    fig=plt.figure()

    C=[1,2,3,4,5,6,7]
    plt.plot(C,train_scores_mean, label="Training set", color="r",linewidth=3,marker='^',markersize=7)
    plt.fill_between(C, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(C, test_scores_mean, label="Validation set", color="g",linewidth=3,marker='D',markersize=7)
    plt.fill_between(C, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.2, color="g")
   
   # plt.set_title()
    plt.xlabel("Sample Number",font2)
    plt.ylabel("Acc",font2)
    plt.xticks(C,[10000,12000,14000,16000,18000,20000,22000,],fontsize=24)
    plt.yticks([0.8,0.85,0.9,0.92,0.94,0.96,0.98,1.0],[0.8,0.85,0.9,0.92,0.94,0.96,0.98,1.0],fontsize=24)
    #plt.yticks([0.8,0.92,0.94,0.96,0.98,1.0],[0.90,0.92,0.94,0.96,0.98,1.0],fontsize=24)
    plt.grid(color='black',linewidth=1,linestyle='-.')
    plt.legend(loc='lower right',fontsize='large',prop=font1)
    plt.title('Model:et(38 channels)       sea',fontsize=30)  
    plt.show()
    

test_learning_curve(train_x,train_y)

