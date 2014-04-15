# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 22:39:56 2014
Desc: Data Normalization
@author: tegjyot
"""
#reading csvfile to array
import numpy as np
filename ='../../dataset/sea_dataset/sea.csv'
data=np.recfromcsv(filename)
print len(data)
print data.dtype


cols=['x1','x2','x3']
for j in cols:
    temp_x=data[j]
    max_x=max(temp_x)
    min_x=min(temp_x)
    data[j]=[float((i-min_x))/(max_x-min_x) for i in temp_x]

np.savetxt("../../dataset/sea_dataset/normalized_sea.csv", data, delimiter=",",header="x1,x2,x3,class", fmt="%s")


