# -*- coding: utf-8 -*-
"""
#plot of accracy per chunk for the differnt .dat files
@author: tegjyot
"""

import simplejson as json
import os
import matplotlib.pyplot as plt

datasets=['em','sea','tj']
folders=['MOA_AUE_NB','MOA_LBHT','traditional_static','PF_LR']
Methodologies=['AUENB','LBHT','traditional','PFLR']
folders=['traditional_static','PF_LR']
Methodologies=['traditional','PFLR']
low_bound=[0.5,0.7,0]
path=os.getcwd()+'/../'
f,axarr= plt.subplots(3,1)
xc=0
chunk_size=1000
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
          	if method=='PF_LR':
		     chunk_accuracy_list=[np.mean(errors[i:i+chunk_size]) for i in range(0,len(errors),chunk_size)]
                else:
		     chunk_accuracy_list=[1-np.mean(errors[i:i+chunk_size]) for i in range(0,len(errors),chunk_size)]
		x=range(0,(len(chunk_accuracy_list)))
           #does not work if , is not specified
           	line,=axarr[xc].plot(x,chunk_accuracy_list)
		lin.append(line)
		axarr[xc].set_title(dataset+' dataset')
          	axarr[xc].set_ylabel('Accuracy')
           	axarr[xc].set_ylim(low_bound[xc],1)
   	xc=xc+1
        
        axarr[2].set_xlabel('Chunk')
        f.legend(tuple(lin),tuple(Methodologies),'upper right')
	f.suptitle('Chunk Based accurac: '+str(chunk_size)+' samples/ chunk')
	
	
