# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 12:38:39 2014

@author: tegjyot
"""

import main_Algo_Geenric as ma
import numpy as np
import math
import matplotlib as mp
from plot_running_avg import plot_running_accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
import pylab
import random

#initialize random number generator
np.random.seed(2000)

filename ='../../dataset/em_dataset/em_dataset.csv'
#filename ='../../dataset/sea_dataset/normalized_sea.csv'
#filename ='../../dataset/tj_stream/tj_stream.csv'
predicted_labels=[]
testing_label=[]
real_labels=[]
predicted_labels_baseline=[]

nop=100#em-100
chunk_size=10#em-10
accuracy_chunking=1000
multiplier=int(float(accuracy_chunking)/chunk_size)
[chunk_accuracy_list,predicted_labels,real_labels] =ma.stub(filename,chunk_size,nop)
mp.pyplot.close()
mp.pyplot.figure()


#performance computation
print(classification_report(real_labels, predicted_labels, target_names=['Class 0','Class1']))
print(confusion_matrix(real_labels, predicted_labels))
print('Accuracy using PF-CD'+str(accuracy_score(real_labels,predicted_labels)))
plot_running_accuracy(real_labels, predicted_labels)


#performance computation with static model
data=np.recfromcsv(filename)
data_tuplelist=data.tolist()
training=int(0.1*len(data_tuplelist))

training_data=[list(i)[:-1] for i in data_tuplelist[:training]]
training_label=[list(i)[-1] for i in data_tuplelist[:training]]

testing_data=[list(i)[:-1] for i in data_tuplelist]
testing_label=[list(i)[-1] for i in data_tuplelist]

lm_model=LogisticRegression(penalty='l2',tol=0.0001, C=1.0, fit_intercept=True)
lm_model.fit(training_data,training_label)
predicted_labels_baseline=lm_model.predict(testing_data)

plot_running_accuracy(testing_label, predicted_labels_baseline)
print('Accuracy using baseline'+str(accuracy_score(testing_label,predicted_labels_baseline)))
mp.pyplot.xlabel('Instance Stream')
mp.pyplot.ylabel('Running Accuracy')
mp.pyplot.title('Running Accuracy Plot(Baseline vs PF-LR)')
mp.pyplot.ylim(0, 1)
mp.pyplot.legend(['PF-LR','Baseline'])
mp.pyplot.show()

mp.pyplot.figure()

#Computing chunk wise accuracy
chunk_accuracy_list_baseline=[np.mod(predicted_labels_baseline[i]+testing_label[i]+1,2) for i in range(0,len(predicted_labels_baseline))]
ma.plot_accuracy(chunk_accuracy_list,multiplier) 
ma.plot_accuracy(chunk_accuracy_list_baseline,accuracy_chunking)
mp.pyplot.xlabel('Chunks')
mp.pyplot.ylabel('Accuracy/Chunk')
mp.pyplot.title(' Accuracy/Chunk (Baseline vs PF-LR)')
mp.pyplot.ylim(0, 1)
mp.pyplot.legend(['PF-LR','Baseline'])
#mp.pyplot.show()
#ma.plot_accuracy(chunk_accuracy_list,multiplier) 