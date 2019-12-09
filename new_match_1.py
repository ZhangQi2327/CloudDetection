# -*- coding: utf-8 -*-
"""
created by ZQ
"""

from netCDF4 import Dataset
import h5py
import numpy as np
import os
from math import sin, asin, cos, radians, fabs, sqrt
#----------------------批量读取GIIRS数据，读出经纬度------------------------------------
landindex=False #True:land False:sea
def haversine(lon1, lat1, lon2, lat2):
        # 经度1，纬度1，经度2，纬度2 （十进制度数）
        # 将十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # 地球平均半径，单位为公里
        return c * r #输出单位为KM
def count(digit,point_g):
        count = 0
        for item in point_g:
            if item == digit:
                count = count + 1
        return count
#a=['4']
a=['49']
#a=['15','16','17','18','19','20','21']
#a=['3','4','5','6','7','8','9','10','11','12','13','14']
#bb=['testdata2']
bb=['testdata1','testdata2']
#sea-test
#CLM_NAME=['20190211190000_20190211191459','20190409024500_20190409025959',
      #    '20190515091500_20190515092959','20190815150000_20190815151459']
#SEA_TRAIN_NEW
#CLM_NAME=['20190207084500_20190207085959','20190130204500_20190130205959',
#         '20190726001500_20190726002959','20190807061500_20190807062959']             
#sea_train...
'''
CLM_NAME=['20190403024500_20190403025959','20190408031500_20190408032959',
          '20190409150000_20190409151459','20190410110000_20190410111459',
          '20190422091500_20190422092959','20190515030000_20190515031459',
          '20190515150000_20190515151459','20190515204500_20190515205959',
          '20190821050000_20190821051459','20190315084500_20190315085959',
          '20190817144500_20190817145959','20190320010000_20190320011459',
          '20190426031500_20190426032959''20190205010000_20190205011459',
          '20190305070000_20190305071459','20190327190000_20190327191459',
          '20190524150000_20190524151459',
          '20190414084500_20190414085959','20190502084500_20190502085959',
          '20190508070000_20190508071459','20190509144500_20190509145959']
'''
#CLM_NAME=['20190508061500_20190508062959','20190603024500_20190603025959',
 #         '20190628084500_20190628085959','20190719090000_20190719091459']
#CLM_NAME=['20190522150000_20190522151459','20190506084500_20190506085959',
 #         '20190220190000_20190220191459','20190207091500_20190207092959',
  #        '20190520024500_20190520025959','20190215144500_20190215145959',
CLM_NAME=['20190210144500_20190210145959']
#land-train
#'20190305061500_20190305062959','20190311121500_20190311122959'
'''CLM_NAME=['20190313001500_20190313002959','20190313121500_20190313122959',
          '20190315121500_20190315122959','20190320001500_20190320002959',
          '20190320121500_20190320122959','20190321061500_20190321062959',
          '20190401001500_20190401002959','20190403121500_20190403122959',
          '20190215061500_20190215062959','20190313061500_20190313062959',
          '20190502121500_20190502122959','20190601061500_20190601062959']
'''
#CLM_NAME=['20190304091500_20190304092959]','20190613061500_20190613062959']
'''
CLM_NAME=['20190201061500_20190201062959','20190215001500_20190215002959',
          '20190601001500_20190601002959','20190701121500_20190701122959',
          '20190124121500_20190124122959','20190415061500_20190415062959',
          '20181115121500_20181115122959']
'''        
#land_test
#CLM_NAME=['20190124001500_20190124002959','20190514121500_20190514122959',
          #'20190515061500_20190515062959','20190615121500_20190615122959']
#CLM_NAME=['20190515061500_20190515062959']    
 


DIR=[]
for k in range(len(a)):
  for x in range(len(bb)):
    DIR.append('D:/fydata/NMC/sea_train/{0}/testdata/{1}/'.format(a[k],bb[x]))
print(DIR)
for k in range(len(a)):
  for x in range(len(bb)):
    mathe=2*k+x
    print('-------------mathe---------------',mathe)
    #number=len([name for name in os.listdir(DIR[mathe]) if os.path.isfile(os.path.join(DIR[mathe], name))])
    number=len([name for name in os.listdir(DIR[mathe]) if name.endswith('.HDF')])
    i=-1
    count_file=0
    lat=np.zeros((number,128))
    lon=np.zeros((number,128))
    LWR=np.ones((689,number*128))
    for name in os.listdir(DIR[mathe]):
       if name.endswith('.HDF'):
        
        count_file+=1
        i=i+1
        print(i)
        f=h5py.File(DIR[mathe]+name,'r')
        lat[i,:]=f.get('IRLW_Latitude')  #每个文件是128个点
        lon[i,:]=f.get('IRLW_Longitude')
        LWR[:,128*i:128*(i+1)]=f.get('ES_RealLW')  #一共是128个点
        f.close
#------------------------提取出想要的GIIRS通道的长波数据，只需要修改index，输出为LWWRR-----------------------
    index=[2,3,5,8,10,11,14,26,32,33,37,40,62,64,69,71,72,74,76,77,78,79,81,82,83,84,85,86,87,88,89,90,110,111,112,280,424,448]
    dd=len(index)
    print('length of index is:',dd)
    LWRR=np.array((dd,number*128))
    LWRR=LWR[index[:],:]
    print(np.shape(LWRR))
    LWWRR=[]

    for i in range(len(LWRR[:,1])): #0:通道 1:GIIRS点数
       for j in range(len(LWRR[1,:])):
         LWWRR.append(str(LWRR[i,j]))
    LWWRR=np.reshape(LWWRR,(dd,number*128))
    del(LWRR)
    
    #-------------
   # print('g_lat',lat)
   # print('g_lon',lon)
    lat=lat.flatten()
    lon=lon.flatten()
    print('lat.shape:',lat.shape)
    print('lon.shape:',lon.shape)
    #-----------------------------------读入入AGRI经纬度-------------------------------
    import scipy.io as sio
    latx = 'D:/fydata/GEO/AGRI_lat.mat'
    lonx= 'D:/fydata/GEO/AGRI_lon.mat'
    laty = sio.loadmat(latx)
    lony = sio.loadmat(lonx)
    AGRI_lat= laty['b_lat']
    AGRI_lon= lony['b_lon']#假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    lat_a=AGRI_lat.astype(np.float16)
    lon_a=AGRI_lon.astype(np.float16)

    del(AGRI_lat)
    del(AGRI_lon)
    #------------------------------------读入cloud_mask-----------------------------------
    #NC和HDF的读取方式相同，都是基于netCDF4
    msk_file = 'D:/fydata/NMC/sea_train/CLM/FY4A-_AGRI--_N_DISK_1047E_L2-_CLM-_MULT_NOM_{}_4000M_V0001.NC'.format(CLM_NAME[k])
    #这里如果直接粘贴复制来的文件路径的话，会出现Errno 22 /xe2/x80/xaa错误，貌似是因为utf-8解码问题
    mask=Dataset(msk_file)
    print('msk_file.variables:',mask.variables.keys())
    clm=mask.variables['CLM'][:]
    clm=np.transpose(clm)  #提取出的Cloud_Mask要转置！！！！！！！
    clm=np.squeeze(clm)
    #------------------------读取地表类型标签--------------
    m_file='D:/fydata/match/land_type.mat'
    land_type = sio.loadmat(m_file)
    qc= land_type['land1']
    #-----------------------从Matlab读出的coast里提取数据-----------------------------
    coast_file='D:/fydata/match/mycoast.mat'
    coast_type1=sio.loadmat(coast_file)
    coast=coast_type1['coast1']
    #--------------------先去掉基本不在GIIRS的经纬度范围的AGRI点（+-0.1扩大边界），不然点太多了------------------------------------
    a_lat=[]
    a_lon=[]
    msk=[]
    land=[]
    a_coast=[]
    latmin=lat.min()
    latmax=lat.max()
    lonmin=lon.min()
    lonmax=lon.max()
    if(latmin<0):
      if(landindex):
         latmin=30
      else:
         latmin=10
    if(lonmin<-180):
       if(landindex):
         lonmin=70
       else:
         lonmin=100
    
    print('latmin:', latmin)
    print('latmax:', latmax)
    print('lonmin:', lonmin)
    print('lonmax:', lonmax)

    for i in range(2748):
        for j in range(2748):
            if(np.logical_and(np.logical_and(lat_a[i,j]>=latmin-0.1,lon_a[i,j]>=lonmin-0.1),np.logical_and(lat_a[i,j]<=latmax+0.1,lon_a[i,j]<=lonmax+0.1))):
                a_lat.append(lat_a[i,j])
                a_lon.append(lon_a[i,j])
                msk.append(clm[i,j])
                land.append(qc[i,j])
                a_coast.append(coast[i,j])
    del(qc)
    del(coast)
    del(lat_a)
    del(lon_a)
    del(clm)
    #！！！！！！！！！！！！！msk是List类型的一定要把它转化成数组类型，不然下面cldmsk.append的时候会出错！！！！！！
    msk=np.array(msk)
    print('type of variable msk is:',type(msk))
    print('-------------------------------------start calculate d between GIIRS and AGRI------------------------------')
    #------------------------以下两种都是获取球面上两点经纬度之间距离的函数，都一样的----------------------
    #EARTH_RADIUS=6371           # 地球平均半径，6371km
    #def hav(theta):
     #   s = sin(theta / 2)
    #    return s * s
    #def get_distance_hav(lat0, lng0, lat1, lng1):
        #"用haversine公式计算球面两点间的距离。"
        # 经纬度转换成弧度
       # lat0 = radians(lat0)
       # lat1 = radians(lat1)
       # lng0 = radians(lng0)
       # lng1 = radians(lng1)
       # dlng = fabs(lng0 - lng1)
       # dlat = fabs(lat0 - lat1)
       # h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
       # distance = 2 * EARTH_RADIUS * asin(sqrt(h))
       # return distance

    #------------------计算AGRI和GIIRS点之间的距离---------------------
    d=np.zeros((len(lat),len(a_lat)),np.float16)
    for i in range(len(lat)):
            for j in range(len(a_lat)):
                d[i,j]=haversine(a_lon[j],a_lat[j],lon[i],lat[i])
               # d[i,j]=get_distance_hav(a_lat[j],a_lon[j],lat[i],lon[i])
    #--------------只留下和每个GIIRS点相距9KM以内的AGRI点-----------------------------------
    point=np.where(d<9)
    point_g=point[0]#存储的是giirs的index，
    point_a=point[1]#存储的是agri的index
    #lat lon 本来就是array数组，所以不需要再转换/
    #！！！！！！！！！a_lon a_lon现在也是list类型，必须转化为array型，然后才能以[A:B,C:D]的方式提取！！！！
    a_lat=np.array(a_lat)
    a_lon=np.array(a_lon)
    point_g=point_g.astype(int)    #point_g 和point_a之后要作为Index，所以需要转化成整数型
    point_a=point_a.astype(int)
    #------------------------------------part end-----------------------
    cldmsk=[]
    land_type=[]
    coast_type=[]

    #-------------------------give cloudmask and truly needed land_type named land_type---------------------------
    #同上，land、a_coast是使用append函数产生的，都是List类型，必须要把land msk a_coast转换成np.array，不然会出现 Only Ingteger scalar can be converted to a scalar index
    land=np.array(land)
    a_coast=np.array(a_coast)
    s=0
    dis=[]
    for i in range(len(lat)):
        a=count(i,point_g)
    #print('type of a ',type(a))
       # print(a)
        b=count(i-1,point_g)
        s=s+b
       # print('s=',s)
        #print('a=',a)
        if i == 0:
            #cldmsk[0,0:a] =
           land_type.append(land[point_a[0:a]])
           cldmsk.append(msk[point_a[0:a]])
           coast_type.append(a_coast[point_a[0:a]])
           dis.append(d[point_g[0:a],point_a[0:a]])
        if i > 0 :
            land_type.append(land[point_a[s:s+a]])
            cldmsk.append(msk[point_a[s :s + a]])
            coast_type.append(a_coast[point_a[s:s+a]])
            dis.append(d[point_g[s:s+a],point_a[s:s+a]])
    del(msk)
    del(land)
    del(a_coast)

    #--------------------------每个GIIRS点的cloud_mask、land_type---------------------------
    g_msk = -np.ones(128*number)
    g_land= -np.ones(128*number)

    ########在这里。每个标签都加上''就是代表是string类型，为了之后顺利写入txt###############
    print('length of cldmsk',len(cldmsk))

    for i in range(len(lat)):

       # print('---------------------------There are too few AGRI FOV for a GIIRS point----------------------------------------')

        count_3 = 0  #每个GIIRS匹配上的AGRI的晴空像元个数
        count_0=0  
        count_1 = 0 # prob cloud
        count_2 = 0 #prob clear  
         #每个GIIRS匹配上的AGRI的cloudy像元个数
        if(len(cldmsk[i])>8):
          for item in cldmsk[i]:
            if item == 3:
                count_3 += 1
            if item ==0:
                count_0+=1
            if item == 1:
                count_1 += 1
            if item == 2:
                count_2 +=2
         
          per_3 = count_3 / len(cldmsk[i])
            # print('per_3',per_3)
          per_0 = count_0 / len(cldmsk[i])
            #print('per_0',per_0)
          per_1 = count_1/ len(cldmsk[i])
           # print('per_1',per_1)
          per_2 = count_2/ len(cldmsk[i])
           # print('per_2',per_2)
        else:
           g_msk[i]=-1
           g_land[i]=-1
           continue
        #--------------give cloudmask to giirs-------------------
        if per_3 > 0.95 :
            g_msk[i]=1 #clear
            continue

            
        if per_0 > 0.95:
            g_msk[i]=0   #cloud
            continue


        newlist=cldmsk[i]
        newdis=dis[i]
        newlist=newlist.tolist()
        newdis=newdis.tolist()
        if 0 in newlist:
           if 3 in newlist:
                g_msk[i]=2


        #------------give land type to giirs---------------------------
    for i in range(len(lat)):
        count_sea=0
        count_land=0
        for item in land_type[i]:
            if item==0:
                count_sea+=1
            if item==8:
                count_land+=1
        if(len(land_type[i]>0)):
          per_sea=count_sea/len(land_type[i])
      #  print('per_sea',per_sea)
          per_land=count_land/len(land_type[i])
        else:
          g_land[i]=-1
          continue
       # print('per_land',per_land)
        if per_sea==1:
            g_land[i]=0  #陆地
        elif per_land==1:
            g_land[i]=1  #海洋
        else:
            g_land[i]=-1 #海岸

    print('shape of g_mask',np.shape(g_msk))
    print('shape of g_land',np.shape(g_land))
    #------------把数据写入txt前，要把每个数据都转化为string类型，-------------------
    print('len(g_msk):',len(g_msk))
    print('type(g_msk):',type(g_msk))

    print('len(g_land):',len(g_land))
    print('type(g_land):',type(g_land))
    lat1=[]
    lon1=[]


    for item in lat:
        lat1.append(str(item))
    for item in lon:
        lon1.append(str(item))
    print(type(lon1))
    del(lat)
    del(lon)


    #------------------------------验证正确-----------------------------------------------------------
    #------------------------------同时将多个变量写入文件writelines------------------------------
    count1=0
    count2=0
    count3=0


    count5=0
    count6=0

    count7=0

    rad1=[]
    rad2=[]
    rad3=[]


    rad5=[]
    rad6=[]
    rad7=[]



    print('------------------len(g_msk):',len(g_msk))
    print('------------------len(g_land):',len(g_land))
  
    with open(DIR[mathe]+'sea_cloud_new.txt','w+') as f1:
        for i in range(len(g_msk)):
            if(g_msk[i]==0 and g_land[i]==0): #海上有云
                count1+=1
                f1.writelines(str(i)+'      '+str(g_land[i])+'      '+str(g_msk[i])+'      '+lat1[i]+'       '+lon1[i]+'      '+LWWRR[0,i]+'    '+LWWRR[1,i]+'    '+LWWRR[2,i]+'    '+LWWRR[3,i]+'    '+LWWRR[4,i]+'    '+LWWRR[5,i]+'    '+LWWRR[6,i]+'    '+LWWRR[7,i]+'    '+LWWRR[8,i]+'    '+LWWRR[9,i]+'    '+LWWRR[10,i]+'    '+LWWRR[11,i]+'    '+LWWRR[12,i]+'    '+LWWRR[13,i]+'    '+LWWRR[14,i]+'    '+LWWRR[15,i]+'    '+LWWRR[16,i]+'    '+LWWRR[17,i]+'    '+LWWRR[18,i]+'    '+LWWRR[19,i]+'    '+LWWRR[20,i]+'    '+LWWRR[21,i]+'    '+LWWRR[22,i]+'    '+LWWRR[23,i]+'    '+LWWRR[24,i]+'    '+LWWRR[25,i]+'    '+LWWRR[26,i]+'    '+LWWRR[27,i]+'    '+LWWRR[28,i]+'    '+LWWRR[29,i]+'    '+LWWRR[30,i]+'    '+LWWRR[31,i]+'    '+LWWRR[32,i]+'    '+LWWRR[33,i]+'    '+LWWRR[34,i]+'    '+LWWRR[35,i]+'    '+LWWRR[36,i]+'    '+LWWRR[37,i]+'\r\n')
                rad1.append(np.squeeze(LWR[:,i]))
    f1.close()
    print('count_sea_cloud_new:',count1)
    with open(DIR[mathe]+'sea_sunny_new.txt','w+') as f2:
        for i in range(len(g_msk)):
            if(g_msk[i]==1 and g_land[i]==0):  #海上晴空
                count2+=1
                f2.writelines(str(i)+'      '+str(g_land[i])+'      '+str(g_msk[i])+'      '+lat1[i]+'       '+lon1[i]+'      '+LWWRR[0,i]+'    '+LWWRR[1,i]+'    '+LWWRR[2,i]+'    '+LWWRR[3,i]+'    '+LWWRR[4,i]+'    '+LWWRR[5,i]+'    '+LWWRR[6,i]+'    '+LWWRR[7,i]+'    '+LWWRR[8,i]+'    '+LWWRR[9,i]+'    '+LWWRR[10,i]+'    '+LWWRR[11,i]+'    '+LWWRR[12,i]+'    '+LWWRR[13,i]+'    '+LWWRR[14,i]+'    '+LWWRR[15,i]+'    '+LWWRR[16,i]+'    '+LWWRR[17,i]+'    '+LWWRR[18,i]+'    '+LWWRR[19,i]+'    '+LWWRR[20,i]+'    '+LWWRR[21,i]+'    '+LWWRR[22,i]+'    '+LWWRR[23,i]+'    '+LWWRR[24,i]+'    '+LWWRR[25,i]+'    '+LWWRR[26,i]+'    '+LWWRR[27,i]+'    '+LWWRR[28,i]+'    '+LWWRR[29,i]+'    '+LWWRR[30,i]+'    '+LWWRR[31,i]+'    '+LWWRR[32,i]+'    '+LWWRR[33,i]+'    '+LWWRR[34,i]+'    '+LWWRR[35,i]+'    '+LWWRR[36,i]+'    '+LWWRR[37,i]+'\r\n')
                rad2.append(np.squeeze(LWR[:,i]))
    f2.close()
    print('count_sea_sunny_new:',count2)
  
    with open(DIR[mathe]+'sea_part_cloud_new.txt','w+') as f3:
        for i in range(len(g_msk)):
            if(g_msk[i]==2 and g_land[i]==0): #海上有云
                count3+=1
                f3.writelines(str(i)+'      '+str(g_land[i])+'      '+str(g_msk[i])+'      '+lat1[i]+'       '+lon1[i]+'      '+LWWRR[0,i]+'    '+LWWRR[1,i]+'    '+LWWRR[2,i]+'    '+LWWRR[3,i]+'    '+LWWRR[4,i]+'    '+LWWRR[5,i]+'    '+LWWRR[6,i]+'    '+LWWRR[7,i]+'    '+LWWRR[8,i]+'    '+LWWRR[9,i]+'    '+LWWRR[10,i]+'    '+LWWRR[11,i]+'    '+LWWRR[12,i]+'    '+LWWRR[13,i]+'    '+LWWRR[14,i]+'    '+LWWRR[15,i]+'    '+LWWRR[16,i]+'    '+LWWRR[17,i]+'    '+LWWRR[18,i]+'    '+LWWRR[19,i]+'    '+LWWRR[20,i]+'    '+LWWRR[21,i]+'    '+LWWRR[22,i]+'    '+LWWRR[23,i]+'    '+LWWRR[24,i]+'    '+LWWRR[25,i]+'    '+LWWRR[26,i]+'    '+LWWRR[27,i]+'    '+LWWRR[28,i]+'    '+LWWRR[29,i]+'    '+LWWRR[30,i]+'    '+LWWRR[31,i]+'    '+LWWRR[32,i]+'    '+LWWRR[33,i]+'    '+LWWRR[34,i]+'    '+LWWRR[35,i]+'    '+LWWRR[36,i]+'    '+LWWRR[37,i]+'\r\n')
                rad3.append(np.squeeze(LWR[:,i]))
    f3.close()
    print('count_sea_part_cloud_new:',count3)

 

    with open(DIR[mathe]+'land_cloud_new.txt','w+') as f5:
        for i in range(len(g_msk)):
            if(g_msk[i]==0 and g_land[i]==1):  #陆地有云
                count5 += 1
                f5.writelines(str(i)+'      '+str(g_land[i])+'      '+str(g_msk[i])+'      '+lat1[i]+'       '+lon1[i]+'      '+LWWRR[0,i]+'    '+LWWRR[1,i]+'    '+LWWRR[2,i]+'    '+LWWRR[3,i]+'    '+LWWRR[4,i]+'    '+LWWRR[5,i]+'    '+LWWRR[6,i]+'    '+LWWRR[7,i]+'    '+LWWRR[8,i]+'    '+LWWRR[9,i]+'    '+LWWRR[10,i]+'    '+LWWRR[11,i]+'    '+LWWRR[12,i]+'    '+LWWRR[13,i]+'    '+LWWRR[14,i]+'    '+LWWRR[15,i]+'    '+LWWRR[16,i]+'    '+LWWRR[17,i]+'    '+LWWRR[18,i]+'    '+LWWRR[19,i]+'    '+LWWRR[20,i]+'    '+LWWRR[21,i]+'    '+LWWRR[22,i]+'    '+LWWRR[23,i]+'    '+LWWRR[24,i]+'    '+LWWRR[25,i]+'    '+LWWRR[26,i]+'    '+LWWRR[27,i]+'    '+LWWRR[28,i]+'    '+LWWRR[29,i]+'    '+LWWRR[30,i]+'    '+LWWRR[31,i]+'    '+LWWRR[32,i]+'    '+LWWRR[33,i]+'    '+LWWRR[34,i]+'    '+LWWRR[35,i]+'    '+LWWRR[36,i]+'    '+LWWRR[37,i]+'\r\n')
                rad5.append(np.squeeze(LWR[:,i]))
    f5.close()
    print('count_land_cloud_new:',count5)

    with open(DIR[mathe]+'land_sunny_new.txt','w+') as f6:
        for i in range(len(g_msk)):
            if(g_msk[i]==1 and g_land[i]==1):  #陆地晴空
                count6+=1
                f6.writelines(str(i)+'      '+str(g_land[i])+'      '+str(g_msk[i])+'      '+lat1[i]+'       '+lon1[i]+'     '+LWWRR[0,i]+'    '+LWWRR[1,i]+'    '+LWWRR[2,i]+'    '+LWWRR[3,i]+'    '+LWWRR[4,i]+'    '+LWWRR[5,i]+'    '+LWWRR[6,i]+'    '+LWWRR[7,i]+'    '+LWWRR[8,i]+'    '+LWWRR[9,i]+'    '+LWWRR[10,i]+'    '+LWWRR[11,i]+'    '+LWWRR[12,i]+'    '+LWWRR[13,i]+'    '+LWWRR[14,i]+'    '+LWWRR[15,i]+'    '+LWWRR[16,i]+'    '+LWWRR[17,i]+'    '+LWWRR[18,i]+'    '+LWWRR[19,i]+'    '+LWWRR[20,i]+'    '+LWWRR[21,i]+'    '+LWWRR[22,i]+'    '+LWWRR[23,i]+'    '+LWWRR[24,i]+'    '+LWWRR[25,i]+'    '+LWWRR[26,i]+'    '+LWWRR[27,i]+'    '+LWWRR[28,i]+'    '+LWWRR[29,i]+'    '+LWWRR[30,i]+'    '+LWWRR[31,i]+'    '+LWWRR[32,i]+'    '+LWWRR[33,i]+'    '+LWWRR[34,i]+'    '+LWWRR[35,i]+'    '+LWWRR[36,i]+'    '+LWWRR[37,i]+'\r\n')
                rad6.append(np.squeeze(LWR[:,i]))
    f6.close()
    print('count_land_sunny_new:',count6)
  
    with open(DIR[mathe]+'land_part_cloud_new.txt','w+') as f7:
        for i in range(len(g_msk)):
            if(g_msk[i]==2 and g_land[i]==1):  #陆地有云
                count7 += 1
                f7.writelines(str(i)+'      '+str(g_land[i])+'      '+str(g_msk[i])+'      '+lat1[i]+'       '+lon1[i]+'      '+LWWRR[0,i]+'    '+LWWRR[1,i]+'    '+LWWRR[2,i]+'    '+LWWRR[3,i]+'    '+LWWRR[4,i]+'    '+LWWRR[5,i]+'    '+LWWRR[6,i]+'    '+LWWRR[7,i]+'    '+LWWRR[8,i]+'    '+LWWRR[9,i]+'    '+LWWRR[10,i]+'    '+LWWRR[11,i]+'    '+LWWRR[12,i]+'    '+LWWRR[13,i]+'    '+LWWRR[14,i]+'    '+LWWRR[15,i]+'    '+LWWRR[16,i]+'    '+LWWRR[17,i]+'    '+LWWRR[18,i]+'    '+LWWRR[19,i]+'    '+LWWRR[20,i]+'    '+LWWRR[21,i]+'    '+LWWRR[22,i]+'    '+LWWRR[23,i]+'    '+LWWRR[24,i]+'    '+LWWRR[25,i]+'    '+LWWRR[26,i]+'    '+LWWRR[27,i]+'    '+LWWRR[28,i]+'    '+LWWRR[29,i]+'    '+LWWRR[30,i]+'    '+LWWRR[31,i]+'    '+LWWRR[32,i]+'    '+LWWRR[33,i]+'    '+LWWRR[34,i]+'    '+LWWRR[35,i]+'    '+LWWRR[36,i]+'    '+LWWRR[37,i]+'\r\n')
                rad7.append(np.squeeze(LWR[:,i]))
    f7.close()
    print('count_land_part_cloud_new:',count7)

    if(count1>1):
        data1=np.loadtxt(DIR[mathe]+'sea_cloud_new.txt')
        data1_info=data1[:,0:5]
        rad1=np.c_[data1_info,rad1]
        np.savetxt(DIR[mathe]+'full_sea_cloud_new.txt',rad1, fmt = '%s')
    else:
        print('count1=0')


    if(count2>1):
        data2=np.loadtxt(DIR[mathe]+'sea_sunny_new.txt')
        data2_info=data2[:,0:5]
        rad2=np.c_[data2_info,rad2]
        np.savetxt(DIR[mathe]+'full_sea_sunny_new.txt',rad2, fmt = '%s')
    else:
        print('count2=0')

    if(count3>1):
        data3=np.loadtxt(DIR[mathe]+'sea_part_cloud_new.txt')
        data3_info=data3[:,0:5]
        rad3=np.c_[data3_info,rad3]
        np.savetxt(DIR[mathe]+'full_sea_part_cloud_new.txt',rad3, fmt = '%s')
    else:
        print('count3=0')






    if(count5>1):
        data5=np.loadtxt(DIR[mathe]+'land_cloud_new.txt')
        data5_info=data5[:,0:5]
        rad5=np.c_[data5_info,rad5]
        np.savetxt(DIR[mathe]+'full_land_cloud_new.txt',rad5, fmt = '%s')
    else:
        print('count5=0')


    if(count6>1):
        data6=np.loadtxt(DIR[mathe]+'land_sunny_new.txt')
        data6_info=data6[:,0:5]
        rad6=np.c_[data6_info,rad6]
        np.savetxt(DIR[mathe]+'full_land_sunny_new.txt',rad6, fmt = '%s')
    else:
        print('count6=0')

  
    if(count7>1):
        data7=np.loadtxt(DIR[mathe]+'land_part_cloud_new.txt')
        data7_info=data7[:,0:5]
        rad7=np.c_[data7_info,rad7]
        np.savetxt(DIR[mathe]+'full_land_part_cloud_new.txt',rad7, fmt = '%s')
    else:
        print('count7=0')



 





