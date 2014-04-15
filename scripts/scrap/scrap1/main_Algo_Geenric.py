# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 16:17:11 2014
Desc:Main algorithm
@author: tegjyot
"""
import numpy as np
import math
import matplotlib as mp

def compute_chunk_n(data,nop,sigmai,old_param,old_index):
    
    batch_size=len(data)
    #number of dimensions
    nod=len(sigmai)
    accuracy_particle=np.zeros(shape=(nop,1))
    current_parameters=np.zeros(shape=(nop,nod))
    
    
    for i in range(0,nop):
        for j in range(0,nod):
            current_parameters[i,j]=np.random.normal(loc=old_index[i,j],scale=sigmai[j])
        accuracy_particle[i]=compute_accuracy(data,current_parameters[i,:])
   
    #appending the previous chunk paramters to list of possible particles
    accuracy_previous=compute_accuracy(data,old_param[0])
    accuracy_particle=np.append(accuracy_particle,accuracy_previous)
    current_parameters=np.append(current_parameters,old_param,axis=0)

    #resampling from the particles which give max accuracy
    #since M>B, there will be more than one particle that gives maximum accuracy
    m=max(accuracy_particle)
    #modified to account for 75% range instead of exact best value float(v)/m>0.75
    max_index=[i for i, v in enumerate(accuracy_particle) if v==m]
    #print(accuracy_particle)
    #print max_index
    new_parameters=np.zeros(shape=(nop,nod))
    
    #resampling based on training accuracy
    for i in range(0,nop):
        particle_sampled=np.random.choice(max_index)
        new_parameters[i]=current_parameters[particle_sampled]
        
    
    #combining parameters using majority voting
    unique_new_params=[list(t) for t in set(map(tuple, new_parameters))]
    chunk_parameters=np.zeros(shape=(1,nod))
    chunk_parameters=[np.mean(i) for i in zip(*unique_new_params)]
        
        
    return [chunk_parameters,new_parameters]


def compute_accuracy(data,current_paramters): 
    accuracy=0
    B=len(data)
    for i in data:
        class_label=i[-1]
        x=i[:-1]
        y=compute_logit(x,current_paramters)
        if (y>=0.5 and class_label==1) or (y<0.5 and class_label==0):
            accuracy=accuracy+1
    return float(accuracy)/B
    
def compute_logit(x,p):
    #print p
    agg=sum(p*q for p,q in zip(x, p))
    y=float(1)/(1+np.exp(-agg))
    return y
 
def compute_predicted(chunk_data,current_parameters):
    predicted_labels_chunk=[]
    for sample in chunk_data:
        y=compute_logit(sample[:-1],current_parameters)
        if (y>=0.5):
            label=1
        else:
            label=0
        predicted_labels_chunk.append(label)
    #print ('hi')  
    print(np.shape(predicted_labels_chunk))
    return predicted_labels_chunk
                
    
def compute_accuracy_stream(chunk_size,nop,nod,sigmai,data):
    old_index=np.random.normal(loc=0,scale=math.pow(sigmai[1],1),size=(nop,nod))
    old_param=np.random.normal(loc=0,scale=sigmai[1],size=(1,nod))
    chunk_accuracy_list=[]
    chunk_predicted_labels=[]
    for i in range(0,len(data),chunk_size):
           print i
           chunk_data=data[i:i+chunk_size]
           chunk_data = [ [1]+x for x in chunk_data]
           chunk_accuracy_list.append(compute_accuracy(chunk_data,old_param[0]))
            #computing predicted labels
           chunk_predicted_labels=chunk_predicted_labels+compute_predicted(chunk_data,old_param[0])
           
           [chunk_params,current_parameters]=compute_chunk_n(chunk_data,nop,sigmai,old_param,old_index)
           old_param=[chunk_params]
           old_index=current_parameters          
    return [chunk_accuracy_list,chunk_predicted_labels] 
    
def plot_accuracy(chunk_accuracy_list,multiplier):
        avg_chunk_accuracy=[]
        n=multiplier
        sum1=0
        for i in range(1,len(chunk_accuracy_list)):
            sum1=sum1+chunk_accuracy_list[i-1]
            if np.mod(i,n)==0:
                avg_chunk_accuracy.append(float(sum1)/n)
                sum1=0

        x=range(1,len(avg_chunk_accuracy)+1)
        #x=range(1,len(chunk_accuracy_list)+1,1000)
        mp.pyplot.plot(x,avg_chunk_accuracy)
        mp.pyplot.show()
        
        	
def stub(filename,chunk_size,nop):
    data=np.recfromcsv(filename)
    data_tuplelist=data.tolist()
    data=[list(i) for i in data_tuplelist]
    real_labels=[sample[-1] for sample in data]
    #subtracted 1 for class field and then added one for  the fields containing all ones for regression coefficient
    nod=np.shape(data)[1]-1+1
    sigmai=[0.1]*nod
    [chunk_accuracy_list,predicted_labels] =compute_accuracy_stream(chunk_size,nop,nod,sigmai,data) 
    return [chunk_accuracy_list,predicted_labels,real_labels] 
   
def main():
    filename ='../../dataset/sea_dataset/normalized_sea.csv'
    nop=100
    chunk_size=50    
    accuracy_chunking=1000
    multiplier=floor(float(accuracy_chunking)/chunk_size)
    [chunk_accuracy_list,chunk_predicted_labels,real_labels] =stub(filename,chunk_size,nop)
    plot_accuracy(chunk_accuracy_list,multiplier) 
    
#    filename ='../../dataset/tj_stream/tj_stream.csv'
#    nop=100
#    chunk_size=50    
#    accuracy_chunking=1000
#    multiplier=floor(float(accuracy_chunking)/chunk_size)
#    [chunk_accuracy_list,chunk_predicted_labels] =stub(filename,chunk_size,nop)
#    plot_accuracy(chunk_accuracy_list,multiplier) 
    

if __name__ == "__main__":
    main()
    
