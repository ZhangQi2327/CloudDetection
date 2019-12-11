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

CLM_NAME=['20190515061500_20190515062959']  #Observation start and end time in the file name of AGRI L2 CLM product   
# a/bb/...HDF     ---->  The path where GIIRS files are stored
a=['1']
bb=['testdata1']  
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
    lat=np.zeros((number,128))  #Each file has 128 field of views
    lon=np.zeros((number,128))
    LWR=np.ones((689,number*128))
    for name in os.listdir(DIR[mathe]):
       if name.endswith('.HDF'):
        count_file+=1
        i=i+1
        print(i)
        f=h5py.File(DIR[mathe]+name,'r')
        lat[i,:]=f.get('IRLW_Latitude')  
        lon[i,:]=f.get('IRLW_Longitude')
        LWR[:,128*i:128*(i+1)]=f.get('ES_RealLW') 
        f.close
    #------------------------------channel index of 38 channels-----------------------
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

    lat=lat.flatten()
    lon=lon.flatten()
    print('lat.shape:',lat.shape)
    print('lon.shape:',lon.shape)
        
    #----------------------------------Read the latitude and longitude of AGRI data-------------------------------
    import scipy.io as sio
    latx = 'D:/fydata/GEO/AGRI_lat.mat'
    lonx= 'D:/fydata/GEO/AGRI_lon.mat'
    laty = sio.loadmat(latx)
    lony = sio.loadmat(lonx)
    AGRI_lat= laty['b_lat']
    AGRI_lon= lony['b_lon']
    lon_a=AGRI_lon.astype(np.float16)
    del(AGRI_lat)
    del(AGRI_lon)

    #------------------------------------Read AGRI L2 product CLM-----------------------------------
    msk_file = 'D:/fydata/NMC/sea_train/CLM/FY4A-_AGRI--_N_DISK_1047E_L2-_CLM-_MULT_NOM_{}_4000M_V0001.NC'.format(CLM_NAME[k])
    mask=Dataset(msk_file)
    print('msk_file.variables:',mask.variables.keys())
    clm=mask.variables['CLM'][:]
    clm=np.transpose(clm)  #提取出的Cloud_Mask要转置！！！！！！！
    clm=np.squeeze(clm)
    #------------------------Read the surface type label of AGRI FOVs--------------
    m_file='D:/fydata/match/land_type.mat'
    land_type = sio.loadmat(m_file)
    qc= land_type['land1']
    #-----------------------Read the land and water boundary label-----------------------------
    coast_file='D:/fydata/match/mycoast.mat'
    coast_type1=sio.loadmat(coast_file)
    coast=coast_type1['coast1']
    #--------------------Remove the AGRI FOVs that are not basically within the range of all FOVs of GIIRS.------------------------------------
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
    
    msk=np.array(msk)
   
    print('-------------------------------------start calculate d between GIIRS and AGRI------------------------------')
    d=np.zeros((len(lat),len(a_lat)),np.float16)
    for i in range(len(lat)):
            for j in range(len(a_lat)):
                d[i,j]=haversine(a_lon[j],a_lat[j],lon[i],lat[i])
    point=np.where(d<9)  #if d<9, the AGRI FOV is saved
    point_g=point[0]
    point_a=point[1]
    a_lat=np.array(a_lat)
    a_lon=np.array(a_lon)
    point_g=point_g.astype(int)  
    point_a=point_a.astype(int)

    cldmsk=[]
    land_type=[]
    coast_type=[]
    land=np.array(land)
    a_coast=np.array(a_coast)
    s=0
    dis=[]
    #--------------get AGRI FOVs for every GIIRS FOV-----------------
    for i in range(len(lat)):
        a=count(i,point_g)
        b=count(i-1,point_g)
        s=s+b
        if i == 0:
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

    #--------------------------calculate the ratio of every cloud label in the GIIRS FOV---------------------------
    g_msk = -np.ones(128*number)
    g_land= -np.ones(128*number)

    print('length of cldmsk',len(cldmsk))

    for i in range(len(lat)):
        count_3 = 0  #clear
        count_0=0  # cloud
        count_1 = 0 # prob cloud
        count_2 = 0 #prob clear  
        if(len(cldmsk[i])>10):
          for item in cldmsk[i]:
            if item == 3:
                count_3 += 1
            if item ==0:
                count_0+=1
            if item == 1:
                count_1 += 1
            if item == 2:
                count_2 +=2
         
          per_3 = count_3 / len(cldmsk[i])  #calculate the ratio of clear AGRI FOV(s) in a GIIRS FOV
          per_0 = count_0 / len(cldmsk[i])
          per_1 = count_1/ len(cldmsk[i])
          per_2 = count_2/ len(cldmsk[i])
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
                g_msk[i]=2   #If the GIIRS FOVs contains clear AGRI FOV(s) and cloud AGRI FOV(s) at the same time, the cloud label of the GIIRS FOV is set to partially cloudy(label=2).


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
          per_land=count_land/len(land_type[i])
        else:
          g_land[i]=-1
          continue

        if per_sea==1:
            g_land[i]=0  #land label
        elif per_land==1:
            g_land[i]=1  #sea  label
        else:
            g_land[i]=-1 #coast  label

    lat1=[]
    lon1=[]
    for item in lat:
        lat1.append(str(item))
    for item in lon:
        lon1.append(str(item))
    print(type(lon1))
    del(lat)
    del(lon)
    #--------------------------------write out the result----------------------------
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
    with open(DIR[mathe]+'sea_cloud_new.txt','w+') as f1:
        for i in range(len(g_msk)):
            if(g_msk[i]==0 and g_land[i]==0): 
                count1+=1
                f1.writelines(str(i)+'      '+str(g_land[i])+'      '+str(g_msk[i])+'      '+lat1[i]+'       '+lon1[i]+'      '+LWWRR[0,i]+'    '+LWWRR[1,i]+'    '+LWWRR[2,i]+'    '+LWWRR[3,i]+'    '+LWWRR[4,i]+'    '+LWWRR[5,i]+'    '+LWWRR[6,i]+'    '+LWWRR[7,i]+'    '+LWWRR[8,i]+'    '+LWWRR[9,i]+'    '+LWWRR[10,i]+'    '+LWWRR[11,i]+'    '+LWWRR[12,i]+'    '+LWWRR[13,i]+'    '+LWWRR[14,i]+'    '+LWWRR[15,i]+'    '+LWWRR[16,i]+'    '+LWWRR[17,i]+'    '+LWWRR[18,i]+'    '+LWWRR[19,i]+'    '+LWWRR[20,i]+'    '+LWWRR[21,i]+'    '+LWWRR[22,i]+'    '+LWWRR[23,i]+'    '+LWWRR[24,i]+'    '+LWWRR[25,i]+'    '+LWWRR[26,i]+'    '+LWWRR[27,i]+'    '+LWWRR[28,i]+'    '+LWWRR[29,i]+'    '+LWWRR[30,i]+'    '+LWWRR[31,i]+'    '+LWWRR[32,i]+'    '+LWWRR[33,i]+'    '+LWWRR[34,i]+'    '+LWWRR[35,i]+'    '+LWWRR[36,i]+'    '+LWWRR[37,i]+'\r\n')
                rad1.append(np.squeeze(LWR[:,i]))
    f1.close()
    print('count_sea_cloud_new:',count1)
    with open(DIR[mathe]+'sea_sunny_new.txt','w+') as f2:
        for i in range(len(g_msk)):
            if(g_msk[i]==1 and g_land[i]==0):  
                count2+=1
                f2.writelines(str(i)+'      '+str(g_land[i])+'      '+str(g_msk[i])+'      '+lat1[i]+'       '+lon1[i]+'      '+LWWRR[0,i]+'    '+LWWRR[1,i]+'    '+LWWRR[2,i]+'    '+LWWRR[3,i]+'    '+LWWRR[4,i]+'    '+LWWRR[5,i]+'    '+LWWRR[6,i]+'    '+LWWRR[7,i]+'    '+LWWRR[8,i]+'    '+LWWRR[9,i]+'    '+LWWRR[10,i]+'    '+LWWRR[11,i]+'    '+LWWRR[12,i]+'    '+LWWRR[13,i]+'    '+LWWRR[14,i]+'    '+LWWRR[15,i]+'    '+LWWRR[16,i]+'    '+LWWRR[17,i]+'    '+LWWRR[18,i]+'    '+LWWRR[19,i]+'    '+LWWRR[20,i]+'    '+LWWRR[21,i]+'    '+LWWRR[22,i]+'    '+LWWRR[23,i]+'    '+LWWRR[24,i]+'    '+LWWRR[25,i]+'    '+LWWRR[26,i]+'    '+LWWRR[27,i]+'    '+LWWRR[28,i]+'    '+LWWRR[29,i]+'    '+LWWRR[30,i]+'    '+LWWRR[31,i]+'    '+LWWRR[32,i]+'    '+LWWRR[33,i]+'    '+LWWRR[34,i]+'    '+LWWRR[35,i]+'    '+LWWRR[36,i]+'    '+LWWRR[37,i]+'\r\n')
                rad2.append(np.squeeze(LWR[:,i]))
    f2.close()
    print('count_sea_sunny_new:',count2)
  
    with open(DIR[mathe]+'sea_part_cloud_new.txt','w+') as f3:
        for i in range(len(g_msk)):
            if(g_msk[i]==2 and g_land[i]==0): 
                count3+=1
                f3.writelines(str(i)+'      '+str(g_land[i])+'      '+str(g_msk[i])+'      '+lat1[i]+'       '+lon1[i]+'      '+LWWRR[0,i]+'    '+LWWRR[1,i]+'    '+LWWRR[2,i]+'    '+LWWRR[3,i]+'    '+LWWRR[4,i]+'    '+LWWRR[5,i]+'    '+LWWRR[6,i]+'    '+LWWRR[7,i]+'    '+LWWRR[8,i]+'    '+LWWRR[9,i]+'    '+LWWRR[10,i]+'    '+LWWRR[11,i]+'    '+LWWRR[12,i]+'    '+LWWRR[13,i]+'    '+LWWRR[14,i]+'    '+LWWRR[15,i]+'    '+LWWRR[16,i]+'    '+LWWRR[17,i]+'    '+LWWRR[18,i]+'    '+LWWRR[19,i]+'    '+LWWRR[20,i]+'    '+LWWRR[21,i]+'    '+LWWRR[22,i]+'    '+LWWRR[23,i]+'    '+LWWRR[24,i]+'    '+LWWRR[25,i]+'    '+LWWRR[26,i]+'    '+LWWRR[27,i]+'    '+LWWRR[28,i]+'    '+LWWRR[29,i]+'    '+LWWRR[30,i]+'    '+LWWRR[31,i]+'    '+LWWRR[32,i]+'    '+LWWRR[33,i]+'    '+LWWRR[34,i]+'    '+LWWRR[35,i]+'    '+LWWRR[36,i]+'    '+LWWRR[37,i]+'\r\n')
                rad3.append(np.squeeze(LWR[:,i]))
    f3.close()
    print('count_sea_part_cloud_new:',count3)

 

    with open(DIR[mathe]+'land_cloud_new.txt','w+') as f5:
        for i in range(len(g_msk)):
            if(g_msk[i]==0 and g_land[i]==1):  
                count5 += 1
                f5.writelines(str(i)+'      '+str(g_land[i])+'      '+str(g_msk[i])+'      '+lat1[i]+'       '+lon1[i]+'      '+LWWRR[0,i]+'    '+LWWRR[1,i]+'    '+LWWRR[2,i]+'    '+LWWRR[3,i]+'    '+LWWRR[4,i]+'    '+LWWRR[5,i]+'    '+LWWRR[6,i]+'    '+LWWRR[7,i]+'    '+LWWRR[8,i]+'    '+LWWRR[9,i]+'    '+LWWRR[10,i]+'    '+LWWRR[11,i]+'    '+LWWRR[12,i]+'    '+LWWRR[13,i]+'    '+LWWRR[14,i]+'    '+LWWRR[15,i]+'    '+LWWRR[16,i]+'    '+LWWRR[17,i]+'    '+LWWRR[18,i]+'    '+LWWRR[19,i]+'    '+LWWRR[20,i]+'    '+LWWRR[21,i]+'    '+LWWRR[22,i]+'    '+LWWRR[23,i]+'    '+LWWRR[24,i]+'    '+LWWRR[25,i]+'    '+LWWRR[26,i]+'    '+LWWRR[27,i]+'    '+LWWRR[28,i]+'    '+LWWRR[29,i]+'    '+LWWRR[30,i]+'    '+LWWRR[31,i]+'    '+LWWRR[32,i]+'    '+LWWRR[33,i]+'    '+LWWRR[34,i]+'    '+LWWRR[35,i]+'    '+LWWRR[36,i]+'    '+LWWRR[37,i]+'\r\n')
                rad5.append(np.squeeze(LWR[:,i]))
    f5.close()
    print('count_land_cloud_new:',count5)

    with open(DIR[mathe]+'land_sunny_new.txt','w+') as f6:
        for i in range(len(g_msk)):
            if(g_msk[i]==1 and g_land[i]==1): 
                count6+=1
                f6.writelines(str(i)+'      '+str(g_land[i])+'      '+str(g_msk[i])+'      '+lat1[i]+'       '+lon1[i]+'     '+LWWRR[0,i]+'    '+LWWRR[1,i]+'    '+LWWRR[2,i]+'    '+LWWRR[3,i]+'    '+LWWRR[4,i]+'    '+LWWRR[5,i]+'    '+LWWRR[6,i]+'    '+LWWRR[7,i]+'    '+LWWRR[8,i]+'    '+LWWRR[9,i]+'    '+LWWRR[10,i]+'    '+LWWRR[11,i]+'    '+LWWRR[12,i]+'    '+LWWRR[13,i]+'    '+LWWRR[14,i]+'    '+LWWRR[15,i]+'    '+LWWRR[16,i]+'    '+LWWRR[17,i]+'    '+LWWRR[18,i]+'    '+LWWRR[19,i]+'    '+LWWRR[20,i]+'    '+LWWRR[21,i]+'    '+LWWRR[22,i]+'    '+LWWRR[23,i]+'    '+LWWRR[24,i]+'    '+LWWRR[25,i]+'    '+LWWRR[26,i]+'    '+LWWRR[27,i]+'    '+LWWRR[28,i]+'    '+LWWRR[29,i]+'    '+LWWRR[30,i]+'    '+LWWRR[31,i]+'    '+LWWRR[32,i]+'    '+LWWRR[33,i]+'    '+LWWRR[34,i]+'    '+LWWRR[35,i]+'    '+LWWRR[36,i]+'    '+LWWRR[37,i]+'\r\n')
                rad6.append(np.squeeze(LWR[:,i]))
    f6.close()
    print('count_land_sunny_new:',count6)
  
    with open(DIR[mathe]+'land_part_cloud_new.txt','w+') as f7:
        for i in range(len(g_msk)):
            if(g_msk[i]==2 and g_land[i]==1): 
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



 





