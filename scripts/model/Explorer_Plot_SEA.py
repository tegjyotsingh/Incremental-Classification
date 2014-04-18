# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 14:37:45 2014
Desc: Initial test scripts for data exploration
Plot SEA dataset
@author: tegjyot
"""

#reading csvfile to array
import numpy as np
filename ='../../dataset/sea_dataset/normalized_sea.csv'
data=np.recfromcsv(filename)
print len(data)
print data.dtype
data_tuplelist=data.tolist()
data_list=[list(i) for i in data_tuplelist]


#visualizing
temp_x=np.vstack((data['x2'],data['x3']))
temp_x.shape
temp_y=data['class']
import matplotlib as mp
import matplotlib.pyplot as plt

#mp.pyplot.scatter(temp_x[0,::100],temp_x[1,::100],s=50,c=temp_y[::100])
plt.figure(1)
plt.suptitle('The SEA Stream')
plt.subplot(221)
plt.scatter(temp_x[0,:15000:10],temp_x[1,:15000:10],s=40,c=temp_y[:15000:10])
plt.xlabel('Stream till 15000')

plt.subplot(222)
plt.scatter(temp_x[0,15000:30000:10],temp_x[1,15000:30000:10],s=40,c=temp_y[15000:30000:10])
plt.xlabel('Stream from 15000-30000')
plt.subplot(223)
plt.scatter(temp_x[0,30000:45000:10],temp_x[1,30000:45000:10],s=40,c=temp_y[30000:45000:10])
plt.xlabel('Stream from 30000-45000')

plt.subplot(224)
plt.scatter(temp_x[0,45000::10],temp_x[1,45000::10],s=40,c=temp_y[45000::10])
plt.xlabel('Stream from 45000-60000')

