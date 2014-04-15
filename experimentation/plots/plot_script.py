# -*- coding: utf-8 -*-
"""
#plot the running accuracy for the different .dat files
@author: tegjyot
"""

import simplejson as json
import os
import matplotlib.pyplot as plt

datasets=['em','sea','tj']
folders=['MOA_AUE_NB','MOA_LBHT','traditional_static','PF_LR']
Methodologies=['AUENB','LBHT','traditional','PFLR']

path=os.getcwd()+'/../'
f,axarr= plt.subplots(1,3)
xc=0

for dataset in datasets:
        lin=[]
	for method in folders:
		#data from PF_LR
		file_path=path+method+'/'+dataset+'.dat'
		json_file=open(file_path)
		json_data=json.load(json_file)
		json_file.close()
		json_data=json_data[0]
           	errors=json_data['error_list']
		#computing running accuracy:
		total=0
		running_accuracy=[]
		for k in range(1,len(errors)+1):
                  total=total+errors[k-1]
                  if method=='PF_LR':
                     running_accuracy.append(float(total)/(k))
                  else:
                     running_accuracy.append(1-float(total)/(k))
		x=range(1,len(errors)+1)
           #does not work if , is not specified
           	line,=axarr[xc].plot(x,running_accuracy)
		lin.append(line)
		axarr[xc].set_title(dataset+' dataset')
           	axarr[xc].set_xlabel('Sample')
          	axarr[xc].set_ylabel('Accuracy')
           	axarr[xc].set_ylim(0.5,1)
   	xc=xc+1
        f.legend(tuple(lin),tuple(Methodologies),'upper right')

	
	
