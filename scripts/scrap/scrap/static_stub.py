
import numpy as np
import math
import matplotlib as mp
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
import pylab
import random  
from scipy import stats

#Script to store output of prediction of traditional model trained on 10% of the initial data stream
#uncomment appropriate filename and dataset
#filename ='../../dataset/em_dataset/em_dataset.csv'
#dataset='em'
#filename ='../../dataset/sea_dataset/normalized_sea.csv'
#dataset='sea'
filename ='../../dataset/tj_stream/tj_stream.csv'
dataset='tj'

predicted_labels=[]
chunk_accuracy_list=[]
testing_label=[]

#parameters
accuracy_chunking=1000
training_percentage=0.1

#performance computation with static model
data=np.recfromcsv(filename)
data_tuplelist=data.tolist()
training=int(0.1*len(data_tuplelist))


training_data=[list(i)[:-1] for i in data_tuplelist[:training]]
training_label=[list(i)[-1] for i in data_tuplelist[:training]]
testing_data=[list(i)[:-1] for i in data_tuplelist]
testing_label=[list(i)[-1] for i in data_tuplelist]

#training Logistic regression on 10% of the initial stream
lm_model=LogisticRegression(penalty='l2',tol=0.0001, C=1.0, fit_intercept=True)
lm_model.fit(training_data,training_label)
predicted_labels=lm_model.predict(testing_data)

#computing accuracy per chunk
error_list=[np.mod(predicted_labels[i]+testing_label[i],2) for i in range(0,len(predicted_labels))]
chunk_accuracy_list=[ float(sum(error_list[i:i+accuracy_chunking]))/accuracy_chunking for i in range(0,len(error_list),accuracy_chunking)]
overall_accuracy=accuracy_score(testing_label,predicted_labels)
print("Accuracy:"+str(overall_accuracy))
#store in JSON file
#store overall_accuracy,chunk_accuracy,predicted labels, dataset,training percentage
op=[{"dataset":dataset, "overall_accuracy": overall_accuracy,
"accuracy_chunk_size":accuracy_chunking, 
"chunk_accuracy":chunk_accuracy_list,  
"error_list":error_list,"training_percentage":training_percentage }]
import simplejson as json
output_file='../../experimentation/traditional_static/'+dataset+'.dat'
f=open(output_file,'w')
json.dump(op,f)
f.close()



