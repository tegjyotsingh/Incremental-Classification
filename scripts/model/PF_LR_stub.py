#Author : Tegjyot Singh Sethi
#Desc:	A stub script to run the  PF_LR algorithm on the three data streams by accessing the main_algo_PF_LR script
#	Thre results are stored in a JSON file

import main_algo_PF_LR as ma
import numpy as np
import math
import matplotlib as mp
from plot_running_avg import plot_running_accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
import pylab
import random  
from scipy import stats

#initialize random number generator
np.random.seed(2234)
predicted_labels=[]
testing_label=[]
real_labels=[]
predicted_labels_baseline=[]

dataset='em'
#read data from file
filename ='../../dataset/em_dataset/em_dataset.csv'
#filename ='../../dataset/sea_dataset/normalized_sea.csv'
#filename ='../../dataset/tj_stream/tj_stream.csv'

data=np.recfromcsv(filename)
data_tuplelist=data.tolist()
data=[list(i) for i in data_tuplelist]
real_labels=[sample[-1] for sample in data]    
    
#specifying parameters
nop=100             #number of particles
chunk_size=10       #number of data samples in a given chunk
accuracy_chunking=1000  #number of data samples to chunk for reporting accuracy
multiplier=floor(float(accuracy_chunking)/chunk_size)

no_of_expts=10
chunk_accuracy=[]
max_accuracy=0
plot_labels=[]
accuracy_list=np.array([])
#compute the values for n iterations of the entire experiemtns
for j in range(0,no_of_expts):
	[chunk_accuracy_list,predicted_labels] =ma.stub(data,chunk_size,nop)
	current_accuracy=accuracy_score(real_labels,predicted_labels)
	if current_accuracy>max_accuracy:
		max_accuracy=current_accuracy
		plot_labels=predicted_labels
		chunk_accuracy=chunk_accuracy_list
	accuracy_list=np.append(accuracy_list,current_accuracy)

#compute statistics:
n1, min_max, mean1, var1, skew, kurt = stats.describe(accuracy_list)
std1=math.sqrt(var1)
R = stats.norm.interval(0.05,loc=mean1,scale=std1)
print('Accuracy using PF-CD at p value of 0.05')
print(R)

error=[np.mod(plot_labels[i]+real_labels[i],2) for i in range(0,len(plot_labels))]

#write output to file
#store Confidence_Interval, chunk_accuracy_values, chunk_size,  predicted_labels,nop,dataset,noofexpts
op=[{"dataset":dataset, "noofexpts": no_of_expts,"accuracy_chunk_size":accuracy_chunking,"chunk_size":chunk_size, "nop":nop,"confidence_interval":list(R), "chunk_accuracy":chunk_accuracy,  "error_list":error}]
import simplejson as json
output_file='../../experimentation/PF_LR/'+dataset+'.dat'
f=open(output_file,'w')
json.dump(op,f)
f.close()



