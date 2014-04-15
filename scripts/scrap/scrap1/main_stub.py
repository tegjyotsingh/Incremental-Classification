# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 17:10:19 2014

@author: tegjyot
"""

import main_Algo_Geenric as ma
import numpy as np
import math
import matplotlib as mp
from plot_running_avg import plot_running_accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
#filename ='../../dataset/sea_dataset/normalized_sea.csv'
#nop=100
#chunk_size=50    
#accuracy_chunking=1000
#multiplier=floor(float(accuracy_chunking)/chunk_size)
#chunk_accuracy_list=ma.stub(filename,chunk_size,nop)
#ma.plot_accuracy(chunk_accuracy_list,multiplier) 
    
filename ='../../dataset/tj_stream/tj_stream.csv'
nop=100
chunk_size=50   
accuracy_chunking=1000
multiplier=int(float(accuracy_chunking)/chunk_size)
[chunk_accuracy_list,predicted_labels,real_labels] =ma.stub(filename,chunk_size,nop)

#performance computation
print(classification_report(real_labels, predicted_labels, target_names=['Class 0','Class1']))
print(confusion_matrix(real_labels, predicted_labels))
plot_running_accuracy(real_labels, predicted_labels)
print('Accuracy using PF-CD'+str(accuracy_score(real_labels,predicted_labels)))

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
predicted_labels=lm_model.predict(testing_data)
plot_running_accuracy(testing_label, predicted_labels)
print('Accuracy using baseline'+str(accuracy_score(testing_label,predicted_labels)))
mp.pyplot.show()
chunk_accuracy_list_baseline=[np.mod(predicted_labels[i]+testing_label[i]+1,2) for i in range(0,len(predicted_labels))]
ma.plot_accuracy(chunk_accuracy_list_baseline,accuracy_chunking)
ma.plot_accuracy(chunk_accuracy_list,multiplier) 